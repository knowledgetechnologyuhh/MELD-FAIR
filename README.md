# MELD-FAIR

This repository contains the realignment data, and the code of the realignment procedure and of the emotional recognition presented at [Whose Emotion Matters? Speaking Activity Localisation without Prior Knowledge](https://www.sciencedirect.com/science/article/pii/S0925231223003946).

The idea of MELD-FAIR is to be a first step towards making the acoustic and visual data of [MELD](https://affective-meld.github.io) more reliable.

MELD is a multimodal multi-party dataset for emotion recognition in conversations. It contains videos with scenes from the *Friends* TV-sitcom as well as transcriptions of the utterances within those scenes.

The textual data of MELD (the transcriptions) are quite reliable. However, this reliability is not reflected in the video scenes. MELD-FAIR was conceived with the aim of addressing this issue.

***

## Dependencies

Install the required packages
```
python3 -m pip install -r requirements.txt
```

Download the [original dataset from MELD](https://affective-meld.github.io) and store it in a new folder.

Extract the source code from TalkNet from [its repository](https://github.com/TaoRuijie/TalkNet-ASD) and store it in a new folder.

Obtain TalkNet's pretrained model `pretrain_AVA.model` and store it in TalkNet's base folder.

**Default folder names are already given in config.py.** If the folder structure differs from the values given in config.py, modify it accordingly.

***

## Generation of MELD-FAIR dataset

The generation of MELD-FAIR dataset is composed of several steps that should take quite a long time each.

### Generation of forced alignment data

The first step consists in generating the CSV with time stamp data for forced alignment and assembling of the realigned videos.

This data associates each video of MELD-FAIR (the realigned version of MELD) with a sequences of chunks of the original videos from MELD.

To execute this step, run the following script at the parent folder:
```
python3 -m MELD-FAIR.realigner.forced_alignment_data_generator
```

In the script above, MELD-FAIR is assumed to be the name of the folder containing this code. In case the name of the folder is different, use the corresponding name.

### Assembling of the videos of MELD-FAIR

In this step, the videos are assembled with ffmpeg based on the information present in the CSV generated in the previous step.

To execute this step, run the following script at the parent folder:
```
python3 -m MELD-FAIR.realigner.realigned_video_assembler
```

### Determination of the regions with faces detected in the MELD-FAIR videos

In this step, for every frame of every video of MELD-FAIR, the delimiting coordinates of the bounding boxes that determine the regions of all detected faces are retrieved.

The bounding boxes retrieved this way are organised into tracks (faces are detected and tracked).

The information of the coordinates of those bounding boxes as well as the tracks they are associated with is then stored in a CSV file.

To execute this step, run the following script at the parent folder:
```
python3 -m MELD-FAIR.asd.face_bbox_determination
```

### Detection of active speakers in the MELD-FAIR videos

In this step, for every video of MELD-FAIR, TalkNet is used to determine the frames in which the face of the active speaker is detected, and to which of the above mentioned tracks it is assigned to.

This step produces another CSV file, which contains the same kind of information that exists in the CSV produced in the previous step.

However, instead of having information regarding all detected faces, it only contains information on the position of the face of the active speaker.

To execute this step, run the following script at the parent folder:
```
python3 -m MELD-FAIR.asd.active_speaker_detection
```

### Extraction of face crops of the active speakers

In this step, for every MELD-FAIR video, faces are cropped given the information given in the CSV produced in the previous step.

To execute this step, run the following script at the parent folder:
```
python3 -m MELD-FAIR.asd.generate_face_crops
```

#### Existing CSV data

A CSV file with bounding box information is already available at the **csvs** folder.

In order to extract face crops using the information in this CSV, remind to modify the content of **config.py** accordingly and run the script above.

***

## Audio-visual emotion recognition with MELD-FAIR dataset

In the folder **er**, there is the implementation of a model for the recognition of emotions from utterances in MELD-FAIR given acoustic and visual data.

***

## Citation

In case you used MELD-FAIR for your research, please cite the following paper:
```
@article{carneiro2023emotion,
	title = {Whose emotion matters? Speaking activity localisation without prior knowledge},
	author = {Hugo Carneiro and Cornelius Weber and Stefan Wermter},
	journal = {Neurocomputing},
	volume = {545},
	pages = {126271},
	year = {2023},
	doi = {10.1016/j.neucom.2023.126271}
}
```
