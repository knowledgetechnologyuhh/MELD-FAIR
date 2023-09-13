import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


from .. import config as cfg


def extract_all_facetracks():
    app = FaceAnalysis(providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id = 0, det_size = (640, 640), det_thresh = 0.5)

    facetracks_df = pd.DataFrame(columns = ["Split", "Dialogue ID", "Utterance ID", "Track ID", "Frame Number", "X Left", "Y Top", "X Right", "Y Bottom"])

    for s in cfg.splits:
        print(f"Generating bound box information from people within the videos of the {s} split.")

        for filename in glob.glob(os.path.join(cfg.meld_realigned_video_folders[s], "**", "*.mp4")):
            base_filename = filename[filename.rfind(os.sep) + 1 : - 4]
            dia_id = int(base_filename[3 : base_filename.rfind("_")])
            utt_id = int(base_filename[base_filename.rfind("_") + 4 :])
            
            print(f"\tDialogue {dia_id} - Utterance {utt_id}")
            
            cap = cv2.VideoCapture(filename)
            fps = cfg.meld_alt_fps if dia_id in cfg.alt_video_prop_dialogues else cfg.meld_main_fps
            
            y_min, x_min = 0, 0
            y_max, x_max = cfg.meld_alt_res if dia_id in cfg.alt_video_prop_dialogues else cfg.meld_main_res
            
            frame_bboxes = {} 
            idx = -1
            while cap.isOpened():
                ret, frame = cap.read()
                idx += 1
                
                if ret == True:
                    faces = app.get(frame)
                    bboxes = np.array([fc.bbox for fc in faces])
                    
                    if bboxes.shape[0] != 0:
                        frame_bboxes[idx] = bboxes
                else:
                    break
            cap.release()
            
            if len(frame_bboxes) == 0: # No face was found throughout the video
                continue
            
            frame_indexes = frame_bboxes.keys()
            min_frame_idx = min(frame_indexes)
            max_frame_idx = max(frame_indexes)
            
            bbox_trk_idx = {}
            tracks = []
            for frame in range(min_frame_idx, max_frame_idx + 1):
                updated_bbox_trk_idx = {}
                if frame - 1 in frame_indexes and frame in frame_indexes:
                    prev_bboxes = frame_bboxes[frame - 1]
                    curr_bboxes = frame_bboxes[frame]
                    
                    ious = {}
                    for idx_bbox0, bbox0 in enumerate(prev_bboxes):
                        for idx_bbox1, bbox1 in enumerate(curr_bboxes):
                            w_bbox0 = bbox0[2] - bbox0[0]
                            h_bbox0 = bbox0[3] - bbox0[1]
                            area_bbox0 = w_bbox0 * h_bbox0
                            
                            w_bbox1 = bbox1[2] - bbox1[0]
                            h_bbox1 = bbox1[3] - bbox1[1]
                            area_bbox1 = w_bbox1 * h_bbox1
                            
                            w_int = max(0, min(bbox0[2], bbox1[2]) - max(bbox0[0], bbox1[0]))
                            h_int = max(0, min(bbox0[3], bbox1[3]) - max(bbox0[1], bbox1[1]))
                            area_int = w_int * h_int
                            
                            ious[(idx_bbox0, idx_bbox1)] = area_int / (area_bbox0 + area_bbox1 - area_int)
                    
                    most_probable_facetracks = [max([(ious[(idx_bbox0, idx_bbox1)], idx_bbox0, idx_bbox1) for idx_bbox0 in range(len(prev_bboxes))]) for idx_bbox1 in range(len(curr_bboxes))]
                    
                    # An IoU threshold of 9 / 23 implies in a variation of at least 25% the size of one of the dimensions of the previous bounding box. 9 / 23 = (3 / 4) ** 2 / (2 - (3 / 4) ** 2)
                    iou_thr = 9 / 23
                    for iou, idx_bbox0, idx_bbox1 in most_probable_facetracks:
                        trk_indexes = [k for k in bbox_trk_idx.keys() if bbox_trk_idx[k] == idx_bbox0]
                        if iou < iou_thr or len(trk_indexes) == 0:
                            tracks.append([])
                            tracks[-1].append((frame, curr_bboxes[idx_bbox1]))
                            updated_bbox_trk_idx[len(tracks) - 1] = idx_bbox1
                        elif len(trk_indexes) == 1:
                            tracks[trk_indexes[0]].append((frame, curr_bboxes[idx_bbox1]))
                            updated_bbox_trk_idx[trk_indexes[0]] = idx_bbox1
                        else:
                            raise Exception("This was not supposed to happen.")
                
                elif frame in frame_indexes:
                    curr_bboxes = frame_bboxes[frame]
                    for idx_bbox, bbox in enumerate(curr_bboxes):
                        tracks.append([])
                        tracks[-1].append((frame, bbox))
                        updated_bbox_trk_idx[len(tracks) - 1] = idx_bbox
                bbox_trk_idx = updated_bbox_trk_idx
            
            for trk_idx, trk in enumerate(tracks):
                for frame_idx, bbox in trk:
                    w_bbox = bbox[2] - bbox[0]
                    h_bbox = bbox[3] - bbox[1]
                    d_bbox = max(h_bbox, w_bbox)
                    
                    x_c = int(np.round((bbox[0] + bbox[2]) / 2))
                    y_c = int(np.round((bbox[1] + bbox[3]) / 2))
                    
                    # Increases the enlarged cropped area by 25% in each dimension, leading the cropped area to be at least 56.25% larger.
                    enlarged_bbox_side_length = int(np.round(0.625 * d_bbox))
                    x0 = x_c - enlarged_bbox_side_length
                    x1 = x_c + enlarged_bbox_side_length
                    y0 = y_c - enlarged_bbox_side_length
                    y1 = y_c + enlarged_bbox_side_length

                    # Translate the bounding box if it transpasses the boundaries of the original video frame.
                    if x0 < x_min:
                        x1 += (x_min - x0)
                        x0 = x_min
                    elif x1 > x_max:
                        x0 += (x_max - x1)
                        x1 = x_max
                    
                    if y0 < y_min:
                        y1 += (y_min - y0)
                        y0 = y_min
                    elif y1 > y_max:
                        y0 += (y_max - y1)
                        y1 = y_max
                    
                    facetracks_df.loc[len(facetracks_df.index)] = [s, dia_id, utt_id, trk_idx, frame_idx, x0, y0, x1, y1]
            
    facetracks_df.to_csv(cfg.facetracks_csv, index = False)
    print(f"Bounding box information on the face of every person of every video has been stored at {cfg.facetracks_csv}.")


if __name__ == "__main__":
    extract_all_facetracks()
