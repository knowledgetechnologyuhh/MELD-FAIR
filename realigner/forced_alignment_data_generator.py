# This script provides a forced alignment between a sequence of transcriptions of utterances from a dialogue and the speech of that dialogue.
# This script is largely based on the one provided in https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
#
# The following modifications were done in order to adequate the script at PyTorch webpage to the problem of realigning the MELD videos:
#   - Concatenation of audios of various utterances into a single audio of the whole dialogue
#   - Similar concatenation of the utterance transcriptions, with the inclusion of start-of-dialogue and end-of-dialogue alongside the start-of-sentence and end-of-sentence tokens already included in the code at the PyTorch webpage
#   - Addition of a method to merge utterances
#
# The method returns CSV files containing start and end time stamps of the time slices of the videos in the original MELD dataset that constitute the realigned video


import os
import sys

import psutil
import shutil
import subprocess
from dataclasses import dataclass

import cv2
import re
import unicodedata
import pickle as pkl
import numpy as np
import pandas as pd
from num2words import num2words

from datetime import datetime, date, time, timedelta

import torch
import torchaudio
import torchaudio.transforms as TF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


from .. import config as cfg
from ..audio import audio_extractor

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Span:
    start: int
    end: int


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class MultiSegment:
    label: str
    spans: list[Span]
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): {[(s.start , s.end) for s in self.spans]}"

    @property
    def length(self):
        return sum([s.end - s.start for s in self.spans])


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


# Merge labels
def merge_repeats(transcript, path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def merge_sentences(word_segments, separator="$", threshold = 50):
    sentences = []
    i1, i2 = 0, 0
    while i1 < len(word_segments):
        if i2 >= len(word_segments) or word_segments[i2].label == separator:
            if i1 + 1 != i2:
                segs = word_segments[i1 + 1:i2]
                sentence = " ".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                
                spans = []
                curr_start = 0
                curr_end = 0
                for i in range(i1 + 1, i2):
                    if curr_start == 0:
                        curr_start = word_segments[i].start
                    if curr_end != 0 and word_segments[i].start - curr_end > threshold:
                        spans.append(Span(max(0, curr_start - (threshold // 6 if len(spans) == 0 else threshold // 2)), curr_end + threshold // 2))
                        curr_start = word_segments[i].start
                    curr_end = word_segments[i].end
                spans.append(Span(max(0, curr_start - (threshold // 6 if len(spans) == 0 else threshold // 2)), curr_end + threshold // 6))
                
                sentences.append(MultiSegment(sentence, spans, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    
    for idx, s in enumerate(sentences):
        if idx < len(sentences) - 1:
            s.spans[-1].end = min(s.spans[-1].end, sentences[idx + 1].spans[0].start - 1)
    return sentences


def format_dialogue(dialogue):
    #print(dialogue)
    transcript = re.sub("[.!\?:…]+", ".", re.sub("[-]+", " ", dialogue))
    #print(transcript)
    thousands_occurrencies = sorted([x.start(0) for x in re.finditer(",[0-9]{3}", transcript)], reverse = True)
    for th in thousands_occurrencies:
        transcript = transcript[: th] + transcript[th + 1 :]
    number_occurrencies = re.findall("[0-9]+", transcript)
    for num in number_occurrencies:
        potential_ordinal = False
        if transcript.index(num) < len(transcript) - len(num) - 1:
            if transcript[transcript.index(num) + len(num) - 1 : transcript.index(num) + len(num) + 2].upper() in ("1ST", "2ND", "3RD") or \
                    transcript[transcript.index(num) + len(num) : transcript.index(num) + len(num) + 2].upper() == "TH":
                potential_ordinal = True
        if potential_ordinal and (len(transcript) == transcript.index(num) + len(num) + 2 or re.match("[^A-Za-z]", transcript[transcript.index(num) + len(num) + 2])):
            transcript = transcript[: transcript.index(num)] + num2words(num, to="ordinal") + transcript[transcript.index(num) + len(num) + 2 :]
        elif transcript.index(num) > 0 and transcript[transcript.index(num) - 1] == "$":
            transcript = transcript[: transcript.index(num) - 1] + num2words(num) + " dollars " + transcript[transcript.index(num) + len(num) :]
        else:
            transcript = transcript[: transcript.index(num)] + num2words(num) + transcript[transcript.index(num) + len(num) :]
    transcript = transcript.replace("%", " percent ").replace("&", " and ").replace("/", "-")
    transcript = re.sub("\|+", "|", re.sub("\\s+", "|", re.sub("[.,;\-\"]", " ", transcript))).upper()
    transcript = "".join(c for c in unicodedata.normalize("NFD", transcript) if unicodedata.category(c) != "Mn")
    transcript = transcript.replace("OK", "OKAY").replace("OKAYAY", "OKAY")
    return transcript


def generate_forced_alignment_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    dictionary = {c: i for i, c in enumerate(labels)}
    
    realignment_df = pd.DataFrame(columns = ["Split", "Dialogue ID", "Utterance ID", "Original Dialogue ID", "Original Utterance ID", "Start Time", "End Time"])
    
    for s in cfg.splits:
        print(f"Split {s.upper()}:")
        
        df = pd.read_csv(cfg.meld_original_csv[s])
        
        for idx, _ in df.iterrows():
            df.loc[df.index[idx], "StartTime"] = (re.sub("^0:", "00:", df.loc[df.index[idx], "StartTime"].replace(",", ".")) + "000")[: 12]
            df.loc[df.index[idx], "EndTime"] = (re.sub("^0:", "00:", df.loc[df.index[idx], "EndTime"].replace(",", ".")) + "000")[: 12]
        
        # Utterance 19 of dialogue 446 of the train split is assigned to the incorrect dialogue. This temporary modification is done for the addition of a new dataframe column named "Corrected Dialogue_ID",
        # which will have the then correctly assigned dialogue (dialogue 447) to that utterance, and the correct ordering of the utterances of that dialogue.
        if s == "train":
            df.loc[df[(df["Dialogue_ID"] == 446) & (df["Utterance_ID"] == 19)].index[0], ["Dialogue_ID", "Utterance_ID", "Episode"]] = [447, 3, 18]
        
        g = df.groupby("Dialogue_ID", group_keys=False)
        df = g.apply(lambda x: x.sort_values(["StartTime", "EndTime", "Utterance_ID"]))
        
        df["Corrected Dialogue_ID"] = df["Dialogue_ID"]
        df["Corrected Utterance_ID"] = 0
        curr_utt_id = 0
        for idx, _ in df.iterrows():
            if idx > 0 and df.loc[df.index[idx], "Dialogue_ID"] == df.loc[df.index[idx - 1], "Dialogue_ID"]:
                curr_utt_id += 1
            else:
                curr_utt_id = 0
            df.loc[df.index[idx], "Corrected Utterance_ID"] = curr_utt_id
        
        # After setting columns "Corrected Dialogue_ID" and "Corrected Utterance_ID", the temporary modification is reversed, but the values in the newly added columns are kept.
        if s == "train":
            df.loc[df[(df["Dialogue_ID"] == 447) & (df["Utterance_ID"] == 3)].index[0], ["Dialogue_ID", "Utterance_ID", "Episode"]] = [446, 19, 19]
        
        df["Time Offset"] = -1
        df["Predicted Time Duration"] = 0
        df["Actual Time Duration"] = 0
        
        for idx, _ in df.iterrows():
            dia_id = df.loc[df.index[idx], "Dialogue_ID"]
            utt_id = df.loc[df.index[idx], "Utterance_ID"]
            
            if dia_id > 0 and dia_id % 100 == 0 and utt_id == 0:
                print(f"The rearranging of the first {dia_id} dialogues of {s} split has been done.")
            
            df.loc[df.index[idx], "Predicted Time Duration"] = int(np.round(cfg.sr * (datetime.combine(date.min, time.fromisoformat(df.loc[df.index[idx], "EndTime"])) - datetime.combine(date.min, time.fromisoformat(df.loc[df.index[idx], "StartTime"]))).total_seconds()))
            
            if (dia_id, utt_id) in cfg.corrupted_utt_videos[s] or (dia_id, utt_id) in cfg.inexistent_utt_videos[s]:
                df.loc[df.index[idx], "Actual Time Duration"] = 0
            else:
                cap = cv2.VideoCapture(cfg.meld_video_filename(s, dia_id, utt_id))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                
                if frame_count != 0.:
                    duration = int(np.round(cfg.sr * frame_count / fps))
                else:
                    duration = 0
                df.loc[df.index[idx], "Actual Time Duration"] = duration
                cap.release()
        
        for idx, _ in df.iterrows():
            if df.loc[df.index[idx], "Corrected Utterance_ID"] == 0:
                df.loc[df.index[idx], "Time Offset"] = 0
            else:
                df.loc[df.index[idx], "Time Offset"] = int(np.round(cfg.sr * (datetime.combine(date.min, time.fromisoformat(df.loc[df.index[idx], "StartTime"])) - datetime.combine(date.min, time.fromisoformat(df.loc[df.index[idx - 1], "EndTime"]))).total_seconds())) + df.loc[df.index[idx], "Predicted Time Duration"] - df.loc[df.index[idx], "Actual Time Duration"]
        
        for idx, _ in df.iterrows():
            if idx < len(df) - 1 and df.loc[df.index[idx], "Dialogue_ID"] == df.loc[df.index[idx + 1], "Dialogue_ID"]:
                if df.loc[df.index[idx + 1], "Time Offset"] < 0:
                    df.loc[df.index[idx], "Actual Time Duration"] += df.loc[df.index[idx + 1], "Time Offset"]
            df.loc[df.index[idx], "Time Offset"] = min(max(0, df.loc[df.index[idx], "Time Offset"]), cfg.sr // 4) # Used to be // 40
        
        for idx, _ in df.iterrows():
            utt = df.loc[df.index[idx], "Utterance"]
            if "(" in utt or "[" in utt:
                parenthesis_depth_level = 0
                brackets_depth_level = 0
                refined_utt = ""
                for ch in utt:
                    if ch == "(":
                        parenthesis_depth_level += 1
                    elif ch == ")":
                        parenthesis_depth_level -= 1
                        if parenthesis_depth_level == 0 and brackets_depth_level == 0:
                            refined_utt += " "
                    elif ch == "[":
                        brackets_depth_level += 1
                    elif ch == "]":
                        brackets_depth_level -= 1
                        if parenthesis_depth_level == 0 and brackets_depth_level == 0:
                            refined_utt += " "
                    elif parenthesis_depth_level == 0 and brackets_depth_level == 0:
                        refined_utt += ch
                    elif parenthesis_depth_level < 0 or brackets_depth_level < 0:
                        raise Exception("\t\tNegative bracket level")
                utt = refined_utt
            df.loc[df.index[idx], "Utterance"] = utt.strip()
        
        df = df.sort_values(["Corrected Dialogue_ID", "Corrected Utterance_ID"])
        
        realignment_data_df = df.filter(["Sr No.", "Utterance", "Speaker", "Emotion", "Dialogue_ID", "Utterance_ID", "Corrected Dialogue_ID", "Corrected Utterance_ID", "Season", "Episode", "StartTime", "EndTime"], axis = 1)
        realignment_data_df.rename(columns={"Dialogue_ID": "Original Dialogue_ID", "Utterance_ID": "Original Utterance_ID", "Corrected Dialogue_ID": "Dialogue_ID", "Corrected Utterance_ID": "Utterance_ID"})
        os.makedirs(os.path.dirname(cfg.meld_realigned_csv[s]), exist_ok = True)
        realignment_data_df.to_csv(cfg.meld_realigned_csv[s], index = False)
        
        print("\tDialogues:")
        for idx_first_utt, _ in df[df["Corrected Utterance_ID"] == 0].iterrows():
            corr_dia_id = df.loc[df.index[idx_first_utt], "Corrected Dialogue_ID"]
            # print(f"\t\tFirst utterance of dialogue {corr_dia_id} - Row {idx_first_utt} of the {s} split CSV data")
            if corr_dia_id > 0 and corr_dia_id % 100 == 0:
                print(f"The proper forced alignment data for the first {corr_dia_id} dialogues of {s} split have been generated.")
            dia_fps = cfg.meld_alt_fps if idx_first_utt in cfg.alt_video_prop_dialogues[s] else cfg.meld_main_fps

            all_dia_ids = df.loc[df["Corrected Dialogue_ID"] == corr_dia_id]["Dialogue_ID"]
            all_utt_ids = df.loc[df["Corrected Dialogue_ID"] == corr_dia_id]["Utterance_ID"]
            all_corr_utt_ids = df.loc[df["Corrected Dialogue_ID"] == corr_dia_id]["Corrected Utterance_ID"]
            all_time_offsets = df.loc[df["Corrected Dialogue_ID"] == corr_dia_id]["Time Offset"]
            all_time_durations = df.loc[df["Corrected Dialogue_ID"] == corr_dia_id]["Actual Time Duration"]

            dialogue_text = "^ " + " $ ^ ".join([df.iloc[idx2]["Utterance"].replace("’", "'").replace("*", "\"").replace("\x91", "'").replace("\x92", "'").replace("\x93", "\"").replace("\x94", "\"").replace("\x96", "-").replace("\x97", "-").replace("—", "-").replace("-", " ") for idx2, _ in df[(df["Corrected Dialogue_ID"] == corr_dia_id)].iterrows()]) + " $"
            transcript = format_dialogue(dialogue_text)
            # print(transcript)
            tokens = [dictionary["-"] if c == "^" or c == "$" else dictionary[c] for c in transcript]
            
            with torch.inference_mode():
                waveform, audio_boundaries = audio_extractor.get_dialogue_waveform(s, corr_dia_id, all_dia_ids, all_utt_ids, all_corr_utt_ids, all_time_offsets, all_time_durations)
                
                # The block below is intended to be used in case the forced alignment data generation procedure ends abruptly due to use of a large amount of virtual memory. 
                """
                if psutil.virtual_memory()[2] > 50:
                    print("\t\t\tToo much RAM usage.")
                    sys.exit(1)
                """
                
                #print(waveform.size())
                emissions, _ = model(waveform.to(device))
                
                # The block below is intended to be used in case the forced alignment data generation procedure ends abruptly due to use of a large amount of virtual memory.
                """
                if psutil.virtual_memory()[2] > 50:
                    print("\t\t\tToo much RAM usage.")
                    sys.exit(1)
                """
                
                emissions = torch.log_softmax(emissions, dim=-1)
            
            emission = emissions[0].cpu().detach()
            trellis = get_trellis(emission, tokens)
            try:
                path = backtrack(trellis, emission, tokens)
            except ValueError as e:
                print(f"\t\t\tFailure during generation of alignment data of dialogue #{corr_dia_id} of split {s}.")
                continue
            segments = merge_repeats(transcript, path)
            word_segments = merge_words(segments)
            ratio = waveform.size(1) / (trellis.size(0) - 1)
            sentence_segments = merge_sentences(word_segments, threshold = cfg.sr / ratio)
            
            if len(sentence_segments) == 0:
                continue
            
            for idx_sent, sentence in enumerate(sentence_segments):
                for sent in sentence.spans:
                    start_at = -1
                    end_at = -1

                    start_ratio = int(np.floor(ratio * sent.start))
                    end_ratio = int(np.ceil(ratio * sent.end))
                    
                    potential_start_videos = [(v, k) for k, v in audio_boundaries.items() if v[0] <= start_ratio]
                    if potential_start_videos == []:
                        start_at = min([(v, k) for k, v in audio_boundaries.items()])
                    else:
                        start_at = max(potential_start_videos)
                    potential_end_videos = [(v, k) for k, v in audio_boundaries.items() if v[1] >= end_ratio]
                    if potential_end_videos == []:
                        end_at = max([(v, k) for k, v in audio_boundaries.items()])
                    else:
                        end_at = min(potential_end_videos)
                    
                    # print(sent, start_at, end_at, start_ratio, end_ratio)
                    
                    start_ts = start_ratio - start_at[0][0] - start_at[0][2] + (0 if start_at[0][3] < 45 * cfg.sr or start_at[0][4] != 0 else (start_at[0][3] - 45 * cfg.sr))
                    end_ts = end_ratio - end_at[0][0] - end_at[0][2] + (0 if start_at[0][3] < 45 * cfg.sr or start_at[0][4] != 0 else (start_at[0][3] - 45 * cfg.sr))
                    
                    # print(start_ts, end_ts)
                    
                    if start_at[0][4] == end_at[0][4]:
                        if end_ts - max(0, start_ts) > 2 * cfg.sr / dia_fps:
                            orig_dia_id, orig_utt_id = df.loc[df[(df["Corrected Dialogue_ID"] == corr_dia_id) & (df["Corrected Utterance_ID"] == start_at[1])].index[0], ["Dialogue_ID", "Utterance_ID"]].values
                            realignment_df.loc[len(realignment_df)] = [s, corr_dia_id, idx_sent, orig_dia_id, orig_utt_id, max(0, start_ts / cfg.sr), end_ts / cfg.sr]
                    else:
                        if start_at[0][3] - max(0, start_ts) > 2 * cfg.sr / dia_fps:
                            orig_dia_id, orig_utt_id = df.loc[df[(df["Corrected Dialogue_ID"] == corr_dia_id) & (df["Corrected Utterance_ID"] == start_at[1])].index[0], ["Dialogue_ID", "Utterance_ID"]].values
                            realignment_df.loc[len(realignment_df)] = [s, corr_dia_id, idx_sent, orig_dia_id, orig_utt_id, max(0, start_ts / cfg.sr), start_at[0][3] / cfg.sr]
                        
                        for k, v in [(k, v) for k, v in audio_boundaries.items() if v[4] > start_at[0][4] and v[4] < end_at[0][4]]:
                            orig_dia_id, orig_utt_id = df.loc[df[(df["Corrected Dialogue_ID"] == corr_dia_id) & (df["Corrected Utterance_ID"] == k)].index[0], ["Dialogue_ID", "Utterance_ID"]].values
                            if (orig_dia_id, orig_utt_id) not in cfg.corrupted_utt_videos[s] and (orig_dia_id, orig_utt_id) not in cfg.inexistent_utt_videos[s] and v[3] > 2 * cfg.sr / dia_fps:
                                realignment_df.loc[len(realignment_df)] = [s, corr_dia_id, idx_sent, orig_dia_id, orig_utt_id, 0, v[3] / cfg.sr]
                        
                        if end_ts > 2 * cfg.sr / dia_fps:
                            orig_dia_id, orig_utt_id = df.loc[df[(df["Corrected Dialogue_ID"] == corr_dia_id) & (df["Corrected Utterance_ID"] == end_at[1])].index[0], ["Dialogue_ID", "Utterance_ID"]].values
                            realignment_df.loc[len(realignment_df)] = [s, corr_dia_id, idx_sent, orig_dia_id, orig_utt_id, 0, end_ts / cfg.sr]
    
    os.makedirs(os.path.dirname(cfg.realignment_timestamps_csv, exist_ok = True)
    realignment_df.to_csv(cfg.realignment_timestamps_csv, float_format = "%.7f", index = False)


if __name__ == "__main__":
    generate_forced_alignment_data()
