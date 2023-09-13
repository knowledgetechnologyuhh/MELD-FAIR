import os
import sys
import glob
import subprocess

import numpy as np
import pandas as pd
import torch
import torchaudio

from .. import config as cfg


def extract_utterance_data(dia, utt, corr_utt, time_offset, time_duration):
    audio_filename = f"{cfg.meld_original_extracted_audio_tmp_folder}/dia{dia}_utt{utt}.wav"
    if not os.path.exists(audio_filename):
        return []
    
    speech_array, _ = torchaudio.load(audio_filename)
    
    speech_full_array = speech_array[:, :]
    if time_duration > 45 * cfg.sr: #A maximum utterance length of 45 seconds. Except for two incorrectly cut scenes in the test set, every other scene is at most 41.04 seconds long.
        if corr_utt == 0:
            speech_full_array = speech_full_array[:, -45 * cfg.sr :]
        else:
            speech_full_array = speech_full_array[:, : 45 * cfg.sr]
    else:
        speech_full_array = speech_array[:, : time_duration]
    if time_offset > 0:
        speech_full_array = torch.cat([torch.zeros(speech_array.size(0), time_offset), speech_full_array], axis = 1) #There is no need for gaps between utterances to be greater than a few milliseconds long. 
    
    return [speech_full_array[0].numpy()]


def get_dialogue_waveform(split, corr_dia, all_dias, all_utts, all_corr_utts, all_time_offsets, all_time_durations):
    list_generated_audio_files = extract_audio(split, corr_dia, zip(all_dias, all_utts), original_meld = True)
    
    audio_boundaries = {}
    offset = 0
    last_end_time = -1
    speech_input_list = []
    
    for dia, utt, corr_utt, time_offset, time_duration in zip(all_dias, all_utts, all_corr_utts, all_time_offsets, all_time_durations):
        for utt_data in extract_utterance_data(dia, utt, corr_utt, time_offset, time_duration):
            if utt_data.shape[0] > 0:
                speech_input_list = np.concatenate([speech_input_list, utt_data])
            audio_boundaries[corr_utt] = (offset, offset + utt_data.shape[0], time_offset, time_duration, corr_utt)
            offset += utt_data.shape[0]
    
    for audio_tmp_filename in list_generated_audio_files:
        os.remove(audio_tmp_filename)
    
    return torch.Tensor(speech_input_list).unsqueeze(0), audio_boundaries


def extract_audio(split, corr_dialogue_id, dia_utt_pairs, original_meld = True):
    list_generated_audio_files = []
    
    video_base_folder = cfg.meld_original_video_folders[split] if original_meld else os.path.join(cfg.meld_realigned_video_folders[split], f"000{corr_dialogue_id}"[-4 :])
    extracted_audio_base_folder = cfg.meld_original_extracted_audio_tmp_folder if original_meld else os.path.join(cfg.meld_realigned_extracted_audio_folders[split], f"000{corr_dialogue_id}"[-4 :])
    os.makedirs(extracted_audio_base_folder, exist_ok = True)
    
    for dia_id, utt_id in dia_utt_pairs:
        commands = ["ffmpeg", "-y", "-i", f"{video_base_folder}/dia{dia_id}_utt{utt_id}.mp4", "-async", "1", "-vn", "-acodec", "pcm_s16le", "-ar", f"{cfg.sr}", f"{extracted_audio_base_folder}/dia{dia_id}_utt{utt_id}.wav"]
        # commands = ["ffmpeg", "-i", f"{video_base_folder}/dia{dia_id}_utt{utt_id}.mp4", f"{extracted_audio_base_folder}/dia{dia_id}_utt{utt_id}.wav"]
        
        retcode = subprocess.call(commands, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
        if retcode == 0:
            list_generated_audio_files.append(f"{extracted_audio_base_folder}/dia{dia_id}_utt{utt_id}.wav")
    
    return list_generated_audio_files
