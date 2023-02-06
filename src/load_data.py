import csv, json, torchaudio, torch, torch.nn.functional
import numpy as np
from torch.utils.data import Dataset

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, hop_ms=10, mfcc=False):
        self.datapath = dataset_json_file   # Path to json with all files to load in the dataset
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.hop_ms=hop_ms  # hopsize
        self.data = data_json['data'] 
        self.audio_conf = audio_conf # Keep the configuration in memory, can be used to add parameters in the future
        self.melbins = self.audio_conf.get('num_mel_bins') # melbins for spec

        # Get all labels
        self.index_dict = {}
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                self.index_dict[row['mid']] = row['index']

        self.label_num = len(self.index_dict)
        self.mfcc = mfcc # If set to true then use mfcc. If false use fbank

    def resample_16k(self, data, sr):
        if(sr == 16000): 
            return data, 16000
        elif(sr == 32000):
            return data[:,::2], 16000
        else:
            raise RuntimeError("Unexpected sampling rate %s" % (sr))

    def cut_or_pad(self, data):
        target_length = self.audio_conf.get('target_length')
        n_frames = data.shape[0]
        dif = target_length - n_frames
        if dif > 0:
            data = torch.nn.functional.pad(data, (0, 0, 0, dif), mode='constant')
        elif dif < 0:
            data = data[0:target_length, :]
        return data

    def __getitem__(self, index):

        infos = self.data[index]

        # Initialize all labels to 0 and put the correct one to 1
        label_indices = np.zeros(self.label_num)
        for label_str in infos['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        # Get the raw wav to get the fbank or the mfcc
        wavraw, sr = torchaudio.load(infos['wav'])
        wavraw, sr = self.resample_16k(wavraw, sr)
        wavraw = wavraw - wavraw.mean()

        # Get fbank or mfcc from torchaudio lib
        if self.mfcc:
            data = torchaudio.compliance.kaldi.mfcc(wavraw, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.hop_ms, num_ceps=128)
        else:
            data = torchaudio.compliance.kaldi.fbank(wavraw, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                     window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.hop_ms)

        # cut if too long or pad if too short, all tensor must be the same size
        data = self.cut_or_pad(data)

        # Normalizing the tensor
        data = (data-torch.mean(data))/torch.std(data)

        return data, label_indices

    def __len__(self):
        return len(self.data)