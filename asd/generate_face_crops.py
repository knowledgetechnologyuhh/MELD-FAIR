import os
import sys

import cv2
import pandas as pd

from .. import config as cfg


def generate_face_crops():
    df = pd.read_csv(cfg.active_speaker_bbox_csv)
    
    for split in df["Split"].unique():
        for dia_id in df[df["Split"] == split]["Dialogue ID"].unique():
            dia_folder = os.path.join(cfg.meld_realigned_extracted_face_folders[split], f"000{dia_id}"[-4 :])
            os.makedirs(dia_folder, exist_ok = True)
            
            for utt_id in df[(df["Split"] == split) & (df["Dialogue ID"] == dia_id)]["Utterance ID"].unique():
                utt_folder = os.path.join(dia_folder, f"0{utt_id}"[-2 :])
                os.makedirs(utt_folder, exist_ok = True)
                
                df_frame_bboxes = df[(df["Split"] == split) & (df["Dialogue ID"] == dia_id) & (df["Utterance ID"] == utt_id)]
                cap = cv2.VideoCapture(os.path.join(cfg.meld_realigned_video_folders[split], f"000{dia_id}"[-4 :], f"dia{dia_id}_utt{utt_id}.mp4"))
                frame_idx = -1
                while cap.isOpened():
                    ret, frame = cap.read()
                    frame_idx += 1
                    
                    if ret == True:
                        df_frame = df_frame_bboxes[df_frame_bboxes["Frame Number"] == frame_idx]
                        if len(df_frame) > 0:                        
                            for _, row in df_frame.iterrows():
                                cv2.imwrite(os.path.join(utt_folder, f"000{frame_idx}"[-4 :] + cfg.face_file_ending), frame[row["Y Top"] : row["Y Bottom"], row["X Left"] : row["X Right"]])
                    else:
                        break
                cap.release()


if __name__ == "__main__":
    generate_face_crops()