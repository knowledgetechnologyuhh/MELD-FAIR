import os
import sys
import shutil
import subprocess

import pandas as pd

from .. import config as cfg
from ..audio import audio_extractor


def extract_videos():
    realignment_df = pd.read_csv(cfg.realignment_timestamps_csv)
    
    for _, df_utt in realignment_df.groupby(["Split", "Dialogue ID", "Utterance ID"]):
        split = df_utt["Split"].values[0]
        dia_id = df_utt["Dialogue ID"].values[0]
        dia_fps = cfg.meld_alt_fps if dia_id in cfg.alt_video_prop_dialogues[split] else cfg.meld_main_fps
        dialogue_folder = os.path.join(cfg.meld_realigned_video_folders[split], f"000{dia_id}"[-4 :])
        utt_id = df_utt["Utterance ID"].values[0]
        realigned_video_name = os.path.join(dialogue_folder, f"dia{dia_id}_utt{utt_id}.mp4")
        
        if dia_id > 0 and dia_id % 20 == 0 and utt_id == 0:
            print(f"The realigned videos corresponding to the first {dia_id} dialogues of {split} split has been assembled.")
        
        os.makedirs(dialogue_folder, exist_ok = True)
        dialogue_tmp_folder = os.path.join(dialogue_folder, "tmp")
        os.makedirs(dialogue_tmp_folder)
        tmp_videolist_filename = os.path.join(dialogue_folder, "meldtmpvideolistfile.txt")
        tmp_videolist_file = open(tmp_videolist_filename, "w")
        
        orig_dia_ids = df_utt["Original Dialogue ID"].values
        orig_utt_ids = df_utt["Original Utterance ID"].values
        start_timestamps = df_utt["Start Time"].values
        end_timestamps = df_utt["End Time"].values
        
        num_tmp_videos = 0
        for odid, ouid, start_ts, end_ts in zip(orig_dia_ids, orig_utt_ids, start_timestamps, end_timestamps):
            raw_video_name = os.path.join(cfg.meld_original_video_folders[split], f"dia{odid}_utt{ouid}.mp4")
            tmp_video_name = os.path.join(dialogue_tmp_folder, f"video_{num_tmp_videos}.mp4")
            # print(split, realigned_video_name, raw_video_name, tmp_video_name, start_ts, end_ts)
            
            commands = ["ffmpeg", "-i", raw_video_name, "-ss", f"{start_ts}", "-to", f"{end_ts}", "-c:v", "libx264", "-ac", "2", "-c:a", "aac", "-b:a", "127k", "-r", f"{dia_fps}", tmp_video_name]#
            
            retcode = subprocess.call(commands, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
            if retcode == 0:
                tmp_videolist_file.write(f"file '{tmp_video_name}'{os.linesep}")
                num_tmp_videos += 1
            else:
                raise Exception(f"\t\t\t\tProblems in the creation of {tmp_video_name} out of {raw_video_name}.")
        tmp_videolist_file.close()
        
        if num_tmp_videos == 0:
            # print("\t\t\t\tNO VIDEO AVAILABLE!")
            os.remove(tmp_videolist_filename)
            shutil.rmtree(dialogue_tmp_folder)
        else:
            commands = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", tmp_videolist_filename, "-c:v", "libx264", "-ac", "2", "-c:a", "aac", "-b:a", "127k", "-r", f"{dia_fps}", realigned_video_name]
            retcode = subprocess.call(commands, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
            
            if retcode == 0:
                os.remove(tmp_videolist_filename)
                shutil.rmtree(dialogue_tmp_folder)
                audio_extractor.extract_audio(split, dia_id, [(dia_id, utt_id)], original_meld = False)
            else:
                raise Exception(f"\t\t\t\tProblems concatenating videos for the composition of {output_video}.")


if __name__ == "__main__":
    extract_videos()
