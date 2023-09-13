import os
import sys
import glob

import cv2
import numpy as np
from scipy.signal import medfilt
import pandas as pd

import torch
import torchaudio
import torchaudio.transforms as TAT
import torchaudio.functional as TAF
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVF

from .. import config as cfg
from ..audio import audio_processor
sys.path.append(cfg.talknet_asd_folder)
from ..TalkNetASD.talkNet import talkNet


def get_asd_scores(asd_model, spec_transform, melscale_transform, tf, device, split, dia_id, utt_id, df_track_frame_bboxes, adjustment_factor = 2.):
    duration_set = [1,1,1,2,2,2,3,3,4,5,6] # In the TalkNet demo, this line was important for a more reliable result -- The set here used is the same of the TalkNet demo. However, there durationSet was a set and here it was modified to be a list.
    
    audio_utterance_filepath = os.path.join(cfg.meld_realigned_extracted_audio_folders[split], f"000{dia_id}"[-4 :], f"dia{dia_id}_utt{utt_id}.wav")
    audio_waveform = audio_processor.load_and_resample_audio_file(audio_utterance_filepath, device, target_sr = cfg.sr)
    audio_feature = audio_processor.extract_MFCC(audio_utterance_filepath, spec_transform, melscale_transform, device)
    
    face_tracks = {trk_idx : ([], []) for trk_idx in df_track_frame_bboxes["Track ID"].unique()}
    
    cap = cv2.VideoCapture(os.path.join(cfg.meld_realigned_video_folders[split], f"000{dia_id}"[-4 :], f"dia{dia_id}_utt{utt_id}.mp4"))
    frame_idx = -1
    while cap.isOpened():
        ret, frame = cap.read()
        frame_idx += 1
        
        if ret == True:
            df_frame = df_track_frame_bboxes[df_track_frame_bboxes["Frame Number"] == frame_idx]
            if len(df_frame) > 0:
                frame = TVF.to_pil_image(frame)
                # Adjustments of sharpness, contrast, and brightness are important in the proper identification of the active speaker.
                # With a low (to no) adjustment some potential speakers receive a negative score from the active speaker detector.
                frame = TVF.adjust_sharpness(frame, adjustment_factor)
                frame = TVF.adjust_contrast(frame, adjustment_factor)
                frame = TVF.adjust_brightness(frame, adjustment_factor)
                
                for _, row in df_frame.iterrows():
                    face_tracks[row["Track ID"]][0].append(frame_idx)
                    face_tracks[row["Track ID"]][1].append(tf(TVF.crop(frame, row["Y Top"], row["X Left"], row["Y Bottom"] - row["Y Top"], row["X Right"] - row["X Left"])).squeeze())
        else:
            break
    cap.release()
    
    all_track_scores = {}
    for trk_id in face_tracks:
        track_video_feature = torch.stack(face_tracks[trk_id][1]).to(device)
        
        length = min((audio_feature.shape[0] - audio_feature.size(0) % 4) / 100, track_video_feature.size(0) / 25)
        if int(np.round(length * 25)) % 25 == 1:
            length -= 1 / 25
        
        track_audio_feature = audio_feature[: int(np.round(length * 100)), :]
        track_video_feature = track_video_feature[: int(np.round(length * 25)), :, :]

        curr_track_scores = [] # Evaluation use TalkNet
        for duration in duration_set:
            batch_size = int(np.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batch_size):
                    inputA = track_audio_feature[i * duration * 100 : (i+1) * duration * 100, :].unsqueeze(0)
                    #print(inputA.shape)
                    inputV = track_video_feature[i * duration * 25 : (i+1) * duration * 25, :, :].unsqueeze(0)
                    #print(inputV.shape)
                    embedA = asd_model.model.forward_audio_frontend(inputA)
                    #print(embedA.shape)
                    embedV = asd_model.model.forward_visual_frontend(inputV)
                    #print(embedV.shape)
                    embedA, embedV = asd_model.model.forward_cross_attention(embedA, embedV)
                    #print(embedA.shape, embedV.shape)
                    out = asd_model.model.forward_audio_visual_backend(embedA, embedV)
                    #print(out.shape)
                    score = asd_model.lossAV.forward(out, labels = None)
                    #print(score.shape)
                    scores.extend(score)
            curr_track_scores.append(scores)
        all_track_scores[trk_id] = np.mean(np.array(curr_track_scores), axis = 0).astype(float)

    all_track_scores = {trk_id : medfilt(all_track_scores[trk_id], min(7, 2 * (((all_track_scores[trk_id].shape[0]) - 1) // 2) + 1)) for trk_id in all_track_scores if all_track_scores[trk_id].shape[0] > 0}
    all_track_scores = {trk_id : {frame_idx : score for frame_idx, score in zip(face_tracks[trk_id][0], all_track_scores[trk_id])} for trk_id in all_track_scores}
    
    return all_track_scores


def select_highest_scoring_tracks(all_track_scores):
    frame_speakers = {}
    selected_tracks = set([])
    for trk_id in all_track_scores:
        for frame_idx in all_track_scores[trk_id]:
            if all_track_scores[trk_id][frame_idx] >= 0:
                if frame_idx not in frame_speakers:
                    frame_speakers[frame_idx] = {}
                frame_speakers[frame_idx][trk_id] = all_track_scores[trk_id][frame_idx]
                selected_tracks.add(trk_id)
    list_frames = sorted(frame_speakers.keys())
    conflicts0 = [sorted(((trk0, len(all_track_scores[trk0])), (trk1, len(all_track_scores[trk1]))), key = lambda x : x[1]) for trk0 in selected_tracks for trk1 in selected_tracks if trk0 < trk1 and len([frm for frm in all_track_scores[trk0] if frm in all_track_scores[trk1]]) > 0]
    conflicts1 = [sorted(((trk0, len([f for f in frame_speakers if max(frame_speakers[f], key = frame_speakers[f].get) == trk0])), (trk1, len([f for f in frame_speakers if max(frame_speakers[f], key = frame_speakers[f].get) == trk1]))), key = lambda x : x[1]) for trk0 in selected_tracks for trk1 in selected_tracks if trk0 < trk1 and len([frm for frm in all_track_scores[trk0] if frm in all_track_scores[trk1]]) > 0]
    
    conflicts1 = [cf for cf in conflicts1 if len([cf2 for cf2 in conflicts1 if cf2[0] == cf[1] and cf2[0][1] < cf2[1][1]]) == 0]
    conflicts1 = [cf for cf in conflicts1] + [(cf[1], cf[0]) for cf in conflicts1 if cf[0][1] == cf[1][1]]
    conflicts1 = [(cf[0][0], cf[1][0]) for cf in conflicts1 if len([cf2 for cf2 in conflicts1 if cf2[0] == cf[0] and cf2[1][1] > cf[1][1] and (cf[1], cf2[1]) in conflicts1]) == 0]
    conflicts1 = {k : [(cf2[1], None if common_speaking_frames == [] else min(common_speaking_frames), None if common_speaking_frames == [] else max(common_speaking_frames)) for cf2 in conflicts1 if cf2[0] == k and (common_speaking_frames := [frm for frm in frame_speakers if k in frame_speakers[frm] and cf2[1] in frame_speakers[frm]])] for k in set([cf[0] for cf in conflicts1])}
    conflicting_tracks = [(trk0, trk1) for trk0 in conflicts1 for trk1 in conflicts1 if trk0 < trk1 and trk1 in [x[0] for x in conflicts1[trk0]] and trk0 in [x[0] for x in conflicts1[trk1]]]

    for cf_trk in conflicting_tracks:
        short_long_seq_pair = [(track_pair[0][0], track_pair[1][0]) for track_pair in conflicts0 if tuple(sorted([track[0] for track in track_pair])) == cf_trk][0]
        conflicts1 = {k : [(track if track != short_long_seq_pair[0] else short_long_seq_pair[1], frm_start, frm_end) for track, frm_start, frm_end in v] for k, v in conflicts1.items() if k != short_long_seq_pair[1]}
    conflicts1 = {k : list(set([(track, min([trk_id[1] for trk_id in v if trk_id[0] == track]), max([trk_id[2] for trk_id in v if trk_id[0] == track])) for track, _, _ in v])) for k, v in conflicts1.items()}

    frame_stats = [(frm, [track if track not in conflicts1 else [trk_id[0] for trk_id in conflicts1[track] if trk_id[1] != None and trk_id[1] <= frm and trk_id[2] != None and trk_id[2] >= frm] for track in frame_speakers[frm]]) for frm in frame_speakers]
    frame_stats = [(frm, [(None if trk_id == [] else trk_id[0]) if isinstance(trk_id, list) else trk_id for trk_id in tracks]) for frm, tracks in frame_stats]
    frame_stats = [(frm, [trk_id for trk_id in tracks if trk_id != None]) for frm, tracks in frame_stats]
    frame_stats = sorted([(frm, tracks[0]) for frm, tracks in frame_stats if len(tracks) > 0])
    
    return frame_stats


def detect_active_speakers():
	asd_model = talkNet()
	asd_model.loadParameters(cfg.talknet_pretrained_model_path)
	device = "cpu" if torch.cuda.is_available() else "cuda"
	asd_model = asd_model.to(device)
	asd_model.model = asd_model.model.to(device)
	asd_model.model = asd_model.model.to(device)
	asd_model.lossAV = asd_model.lossAV.to(device)
	asd_model.eval()

	tf = TVT.Compose([
		TVT.Grayscale(),
		TVT.Resize((224, 224)),
		TVT.CenterCrop(112),
		TVT.ToTensor(),
		TVT.Normalize((0,), (1 / 255,))
	])

	spec_transform = TAT.Spectrogram(
		n_fft = cfg.n_fft,
		win_length = int(cfg.win_length * cfg.sr),
		hop_length = int(cfg.winstep * cfg.sr),
		window_fn = torch.ones
	).to(device)

	melscale_transform = TAT.MelScale(
		n_mels = cfg.n_mels,
		sample_rate = cfg.sr,
		n_stft = cfg.n_fft // 2 + 1,
		mel_scale = "htk"
	).to(device)

	df_allfacetracks = pd.read_csv(cfg.facetracks_csv)
	df_active_speaker_facetracks = pd.DataFrame(columns = ["Split", "Dialogue ID", "Utterance ID", "Frame Number", "X Left", "Y Top", "X Right", "Y Bottom"])
	
	for split in df_allfacetracks["Split"].unique():
		for dia_id in df_allfacetracks[df_allfacetracks["Split"] == split]["Dialogue ID"].unique():
			for utt_id in df_allfacetracks[(df_allfacetracks["Split"] == split) & (df_allfacetracks["Dialogue ID"] == dia_id)]["Utterance ID"].unique():
				df_track_frame_bboxes = df_allfacetracks[(df_allfacetracks["Split"] == split) & (df_allfacetracks["Dialogue ID"] == dia_id) & (df_allfacetracks["Utterance ID"] == utt_id)]
				face_track_scores = get_asd_scores(asd_model, spec_transform, melscale_transform, tf, device, split, dia_id, utt_id, df_track_frame_bboxes, adjustment_factor = 2.)
				frame_track_association = select_highest_scoring_tracks(face_track_scores)
				
				# print(f"{split.upper()} - ({dia_id}, {utt_id}): {len(frame_track_association)} out of {len(df_track_frame_bboxes['Frame Number'].unique())} frames with associated active speaker (out of {len(df_track_frame_bboxes['Track ID'].unique())} tracks with potential active speakers).")
				for frame, track in frame_track_association:
					x_left, y_top, x_right, y_bottom = df_track_frame_bboxes[(df_track_frame_bboxes["Track ID"] == track) & (df_track_frame_bboxes["Frame Number"] == frame)][["X Left", "Y Top", "X Right", "Y Bottom"]].values[0]
					# print(f"\tFrame {frame} - Track {track}: Coords {(x_left, y_top, x_right, y_bottom)}")
					
					df_active_speaker_facetracks.loc[len(df_active_speaker_facetracks)] = [split, dia_id, utt_id, frame, x_left, y_top, x_right, y_bottom]
				# print()
	
	df_active_speaker_facetracks.to_csv(cfg.active_speaker_bbox_csv, index = False)


if __name__ == "__main__":
    detect_active_speakers()