import os
import sys

# Base path of the whole project
base_folder = os.path.dirname(os.path.abspath(__file__))
# Folder in which data from MELD are to be stored (both original and realigned versions)
meld_folder = os.path.join(base_folder, "MELD")
# Folder where the original MELD data are stored
meld_raw_folder = os.path.join(meld_folder, "MELD.Raw")
# Folders where the videos of the original version of MELD are stored
meld_original_video_folders = {
                                "train" : os.path.join(meld_raw_folder, "train_splits"),
                                "dev"   : os.path.join(meld_raw_folder, "dev_splits_complete"),
                                "test"  : os.path.join(meld_raw_folder, "output_repeated_splits_test")
                              }
# List of identifications (split, dialogue id, utterance id) of original MELD videos whose files are corrupted
corrupted_utt_videos = {"train" : [(125, 3)], "dev" : [], "test" : []}
# List of identifications (split, dialogue id, utterance id) of CSV rows from original MELD which do not have a corresponding video
inexistent_utt_videos = {"train" : [], "dev" : [(110, 7)], "test" : []}

# List of splits (train, dev, and test)
splits = meld_original_video_folders.keys()
# Function the path of a video from MELD's original version given the split, dialogue id, and utterance id
meld_video_filename = lambda split, dia_id, utt_id : os.path.join(meld_original_video_folders[split], f"dia{dia_id}_utt{utt_id}.mp4")
# Path of the CSV with the data of a given split of the original MELD dataset
meld_original_csv = {s : os.path.join(meld_raw_folder, f"{s}_sent_emo.csv") for s in splits}

# Parameters for audio processing
n_fft = 512
win_length = 0.025
winstep = 0.01
n_mels = 26
n_mfcc = 13
sr = 16_000
# Folder to store temporary wave files extracted from the videos of the original MELD
meld_original_extracted_audio_tmp_folder = os.path.join(meld_folder, "original_audio_tmp")

# FPS rate of the MELD videos
meld_main_fps = 24_000 / 1_001 # Most videos in MELD use NTSC standard frame rate of 24000/1001, which is roughly 23.98 fps.
meld_alt_fps = 25. # A few videos use a frame rate of 25 fps.
# Resolution of the MELD videos
meld_main_res = (720, 1080) # (height, width)
meld_alt_res = (384, 496)
# List of identifications (split, dialogue id) of MELD videos with a resolution of 496x384 and sampled at 25 fps 
alt_video_prop_dialogues = {"train" : [128, 203, 383, 517, 710, 967], "dev" : [88], "test" : [184]} # Dialogues whose videos properties are not the ones common to most of MELD videos, but an alternative variation

# Folder where the data of realigned version of MELD is stored
meld_realigned_folder = os.path.join(meld_folder, "realigned")
# Path of the CSV with the data of a given split of the realigned MELD
meld_realigned_csv = {s : os.path.join(meld_realigned_folder, f"realigned_{s}_sent_emo.csv") for s in splits}

# Path of the CSV with the realignment timestamps
realignment_timestamps_csv = os.path.join(meld_realigned_folder, "MELD_video_realignment_timestamps.csv")
# Path of the CSV with the information of the bounding box of every face captured in each video of realigned MELD
facetracks_csv = os.path.join(meld_realigned_folder, "MELD_all_faces_bboxes_and_tracks.csv")
# Path of the CSV with the information of the bounding box of the face of the person identified as the active speaker in each video of realigned MELD
active_speaker_bbox_csv = os.path.join(meld_realigned_folder, "MELD_active_speaker_face_bboxes.csv")

# Subfolder with data of a particular split from realigned MELD
meld_realigned_split_folders = {s : os.path.join(meld_realigned_folder, s) for s in splits}
# Subfolder with videos of a particular split from realigned MELD
meld_realigned_video_folders = {s : os.path.join(meld_realigned_split_folders[s], "videos") for s in splits}
# Subfolder with the audio extracted from the videos of realigned MELD
meld_realigned_extracted_audio_folders = {s : os.path.join(meld_realigned_split_folders[s], "audio", f"{sr}") for s in splits}
# Subfolder with the face crops of active speakers from the videos of realigned MELD
meld_realigned_extracted_face_folders = {s : os.path.join(meld_realigned_split_folders[s], "faces") for s in splits}
# Ending of the files containing the face crops of the active speakers
face_file_ending = ".jpg"

# Folder containing the source code of TalkNet, which should be extracted from its repository at https://github.com/TaoRuijie/TalkNet-ASD
talknet_asd_folder = os.path.join(base_folder, "TalkNetASD")
# Path of the pretrained model of TalkNet used to detect active speakers within the videos of realigned MELD. To obtain the pretrained model of TalkNet, please refer to its repository
talknet_pretrained_model_path = os.path.join(talknet_asd_folder, "pretrain_AVA.model")