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

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
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


def merge_repeats(path):
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
    transcript = re.sub('[.!\?:…]+', '.', re.sub('[-]+', ' ', dialogue))
    #print(transcript)
    thousands_occurrencies = sorted([x.start(0) for x in re.finditer(',[0-9]{3}', transcript)], reverse = True)
    for th in thousands_occurrencies:
        transcript = transcript[: th] + transcript[th + 1 :]
    number_occurrencies = re.findall('[0-9]+', transcript)
    for num in number_occurrencies:
        potential_ordinal = False
        if transcript.index(num) < len(transcript) - len(num) - 1:
            if transcript[transcript.index(num) + len(num) - 1 : transcript.index(num) + len(num) + 2].upper() in ('1ST', '2ND', '3RD') or \
                    transcript[transcript.index(num) + len(num) : transcript.index(num) + len(num) + 2].upper() == 'TH':
                potential_ordinal = True
        if potential_ordinal and (len(transcript) == transcript.index(num) + len(num) + 2 or re.match('[^A-Za-z]', transcript[transcript.index(num) + len(num) + 2])):
            transcript = transcript[: transcript.index(num)] + num2words(num, to='ordinal') + transcript[transcript.index(num) + len(num) + 2 :]
        elif transcript.index(num) > 0 and transcript[transcript.index(num) - 1] == '$':
            transcript = transcript[: transcript.index(num) - 1] + num2words(num) + ' dollars ' + transcript[transcript.index(num) + len(num) :]
        else:
            transcript = transcript[: transcript.index(num)] + num2words(num) + transcript[transcript.index(num) + len(num) :]
    transcript = transcript.replace('%', ' percent ').replace('&', ' and ').replace('/', '-')
    transcript = re.sub('\|+', '|', re.sub('\\s+', '|', re.sub('[.,;\-"]', ' ', transcript))).upper()
    transcript = ''.join(c for c in unicodedata.normalize('NFD', transcript) if unicodedata.category(c) != 'Mn')
    transcript = transcript.replace('OK', 'OKAY').replace('OKAYAY', 'OKAY')
    return transcript

def extract_utterance_data(audio_base_folder, dia, utt, corr_utt, time_offset, time_duration, sr = 16_000):
    audio_filename = os.path.join(audio_base_folder, ('0' * 3 + f'{dia}')[-4 :], f'dia{dia}_utt{utt}.wav')
    if not os.path.exists(audio_filename):
        return []
    
    speech_array, sr = torchaudio.load(audio_filename)
    
    speech_full_array = speech_array[:, :]
    if time_duration > 45 * sr: #A maximum utterance length of 45 seconds. Except for two incorrectly cut scenes in the test set, every other scene is at most 41.04 seconds long.
        if corr_utt == 0:
            speech_full_array = speech_full_array[:, -45 * sr :]
        else:
            speech_full_array = speech_full_array[:, : 45 * sr]
    else:
        speech_full_array = speech_array[:, : time_duration]
    if time_offset > 0:
        speech_full_array = torch.cat([torch.zeros(speech_array.size(0), time_offset), speech_full_array], axis = 1) #There is no need for gaps between utterances to be greater than a few milliseconds long. 
    
    return [speech_full_array[0].numpy()]

def get_dialogue_waveform(audio_base_folder, dia, all_utts, all_corr_utts, all_time_offsets, all_time_durations):
    audio_boundaries = {}
    offset = 0
    last_end_time = -1
    speech_input_list = []
    for utt, corr_utt, time_offset, time_duration in zip(all_utts, all_corr_utts, all_time_offsets, all_time_durations):
        for utt_data in extract_utterance_data(audio_base_folder, dia, utt, corr_utt, time_offset, time_duration):
            if utt_data.shape[0] > 0:
                speech_input_list = np.concatenate([speech_input_list, utt_data])
            audio_boundaries[utt] = (offset, offset + utt_data.shape[0], time_offset, time_duration, corr_utt)
            offset += utt_data.shape[0]
    return torch.Tensor(speech_input_list).unsqueeze(0), audio_boundaries
