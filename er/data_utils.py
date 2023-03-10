import os
import glob
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as TAT
import torchaudio.functional as TAF
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVF


def get_spec_and_melscale_transforms(n_fft, win_length, winstep, n_mels, sample_rate):
    spec_transform = TAT.Spectrogram(
        n_fft = n_fft,
        win_length = int(win_length * sample_rate),
        hop_length = int(winstep * sample_rate),
        window_fn = torch.ones
    )

    melscale_transform = TAT.MelScale(
        n_mels = n_mels,
        sample_rate = sample_rate,
        n_stft = n_fft // 2 + 1,
        mel_scale = 'htk'
    )
    
    return spec_transform, melscale_transform


def extract_MFCC(audio, spec_transform, melscale_transform):
    audio = audio / torch.max(audio)
    audio = audio * 2 ** 20
    coeff = 0.97
    audio = torch.cat((audio[0].unsqueeze(0), audio[1:] - coeff * audio[:-1])) # preemphasis
    spec = spec_transform(audio)
    energy = torch.sum(spec, axis = 0)
    melspec = melscale_transform(spec)
    log_offset = 1e-6
    melspec = torch.log(melspec + log_offset)
    n_mfcc = 13
    dct_mat = TAF.create_dct(n_mfcc, melscale_transform.n_mels, 'ortho')
    mfcc = torch.matmul(melspec.transpose(-1, -2), dct_mat).transpose(-1, -2)
    mfcc[0] = torch.log(energy + log_offset) * 2
    
    return mfcc.transpose(0, 1)


def collate_data(data, split, num_classes, video = True, audio = True, min_num_frames = 15, target_sr = 16_000):
    face_seq_list = []
    speech_list = []
    
    if video:
        size = 112
        
        tf = TVT.Compose([
            TVT.Resize((size, size)),
            TVT.ToTensor()
        ])
        
        for entry in data:
            if split == 'train':
                # Randomly select a part within the video
                offset = torch.randint(max(min_num_frames, len(entry['video_paths'])) - min_num_frames + 1, (1,)).numpy()[0]
            else:
                # Select the central part of the video
                offset = (max(min_num_frames, len(entry['video_paths'])) - min_num_frames) // 2
            
            face_seq = []
            for idx, facepath in enumerate(entry['video_paths']):
                if idx < offset or idx >= offset + min_num_frames:
                    continue
                
                face = Image.open(facepath)
                face = tf(face)
                face_seq.append(face)
            face_seq = [face_seq[0] for _ in range(int(np.floor((min_num_frames - len(face_seq)) / 2)))] + face_seq + [face_seq[-1] for _ in range(int(np.ceil((min_num_frames - len(face_seq)) / 2)))]
            face_seq = torch.stack(face_seq)
            
            if split == 'train':
                urng = torch.rand(4).numpy()
                if urng[0] * 4 < 1:
                    face_seq = TVF.hflip(face_seq)
                elif urng[0] * 4 < 2:
                    transl_offset = int(size * (0.7 + 0.3 * urng[1]))
                    xc, yc = int((size - transl_offset) * urng[2]), int((size - transl_offset) * urng[3])
                    face_seq = face_seq[:, :, yc : yc + transl_offset, xc : xc + transl_offset]
                    face_seq = TVF.resize(face_seq, (size, size))
                elif urng[0] * 4 < 3:
                    face_seq = TVF.rotate(face_seq, 30 * urng[1] - 15)
            face_seq_list.append(face_seq)
    
    if audio:
        audio_files_per_label = [[entry['audio_path'] for entry in data if entry['label'] == label] for label in range(num_classes)]
        
        raw_audio_files = {filename : torchaudio.load(filename) for filename in set([entry['audio_path'] for entry in data])}
        raw_audio_files = {filename : TAF.resample(raw_audio_files[filename][0], raw_audio_files[filename][1], target_sr)[0] for filename in raw_audio_files}
        
        amp_to_db = TAT.AmplitudeToDB('magnitude')
        urng_audio = torch.rand(len(data), 2)
        for rng, entry in zip(urng_audio, data):
            audio_signal = raw_audio_files[entry['audio_path']].detach().clone()
            
            if split == 'train' and len(audio_files_per_label[entry['label']]) > 1 and rng[0] > 0.5:
                snr = torch.rand(1) * 10 - 5
                
                overlapping_audio_slices = [filename for filename in audio_files_per_label[entry['label']] if filename != entry['audio_path']]
                
                if len(overlapping_audio_slices) > 0:
                    overlapping_audio_data = overlapping_audio_slices[int(rng[1] * len(overlapping_audio_slices))]
                    interfering_signal = raw_audio_files[overlapping_audio_data].detach().clone()
                     
                    shortage = audio_signal.size(0) - interfering_signal.size(0)
                    if shortage <= interfering_signal.size(0):
                        if shortage > 0:
                            interfering_signal = F.pad(interfering_signal.unsqueeze(0).unsqueeze(0), (0, shortage), 'circular').squeeze()
                        else:
                            interfering_signal = interfering_signal[: audio_signal.size(0)]

                        cleanDB = amp_to_db(audio_signal)
                        noiseDB = amp_to_db(interfering_signal)
                        interfering_signal = TAF.DB_to_amplitude(cleanDB - noiseDB - snr, ref = torch.mean(interfering_signal), power = 0.5)

                        audio_signal += interfering_signal
            
            speech_list.append(audio_signal)
        
        n_fft = 512
        win_length = 0.025
        winstep = 0.01
        n_mels = 13

        spec_transform, melscale_transform = get_spec_and_melscale_transforms(n_fft, win_length, winstep, n_mels, target_sr)
        
        avg_audio_size = int(np.mean([x.size(0) for x in speech_list]))
        for idx, audio_data in enumerate(speech_list):
            if len(speech_list) > 1:
                while True:
                    shortage = avg_audio_size - audio_data.size(0)
                    if shortage <= 0:
                        if shortage < 0:
                            if split == 'train':
                                offset = int(rng[0] * (1 - shortage))
                            else:
                                offset = -shortage // 2
                            audio_data = audio_data[offset : offset + avg_audio_size]
                        break
                    else:
                        audio_data = F.pad(audio_data.unsqueeze(0).unsqueeze(0), (0, min(shortage, audio_data.size(0))), 'circular').squeeze()
            speech_list[idx] = extract_MFCC(audio_data, spec_transform, melscale_transform)
    
    if video and audio:
        return torch.stack(face_seq_list), torch.stack(speech_list), torch.LongTensor([entry['dialogue'] for entry in data]), torch.LongTensor([entry['utterance'] for entry in data]), torch.LongTensor([entry['label'] for entry in data])
    elif video:
        return torch.stack(face_seq_list), torch.LongTensor([entry['dialogue'] for entry in data]), torch.LongTensor([entry['utterance'] for entry in data]), torch.LongTensor([entry['label'] for entry in data])
    elif audio:
        return torch.stack(speech_list), torch.LongTensor([entry['dialogue'] for entry in data]), torch.LongTensor([entry['utterance'] for entry in data]), torch.LongTensor([entry['label'] for entry in data])
    else:
        raise NotImplementedError


class MELDDataset(Dataset):
    def __init__(self, meld_folder : str, label_list : list, split : str) -> None:
        self.meld_folder = meld_folder
        self.split = split
        
        with pd.read_csv(os.path.join(self.meld_folder, 'csvs', 'MELD_active_speaker_face_bboxes.csv')) as df_face_info:
            utterances_with_face_info = df_face_info[df_face_info['Split'] == f'{self.split}'].groupby(['Dialogue ID', 'Utterance ID']).size().reset_index()[['Dialogue ID', 'Utterance ID']].itertuples(index = False)

        with pd.read_csv(os.path.join(self.meld_folder, 'csvs', f'realigned_{split}_sent_emo.csv')) as df:
            self.data = []
            for idx, _ in df.iterrows():
                dia_id = df.loc[df.index[idx], 'Dialogue_ID']
                utt_id = df.loc[df.index[idx], 'Utterance_ID']
                
                label_id = label_list.index(df.loc[df.index[idx], 'Emotion'])
                
                if (dia_id, utt_id) in utterances_with_face_info:
                    audio_filepath = os.path.join(self.meldfolder, 'MELD_av_data', f'{self.split}', 'audio', f'000{dia_id}'[-4 :], f'dia{dia_id}_utt{utt_id}.wav')
                    video_filepaths = sorted(glob.glob(os.path.join(self.meldfolder, 'MELD_av_data', f'{self.split}', 'faces', f'000{dia_id}'[-4 :], f'0{corr_utt_id}'[-2 :], '*.png')))
                    
                    self.data.append((dia_id, utt_id, label_id, audio_filepath, video_filepaths))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index : int) -> dict:
        dia, utt, label, audio_filepath, video_filepaths = self.data[index]
                
        data = {}
        data['dialogue'] = dia
        data['utterance'] = utt
        data['label'] = label
        data['audio_path'] = audio_filepath
        data['video_paths'] = video_filepaths
        
        return data
