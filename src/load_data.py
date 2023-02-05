import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import os
import gc
import librosa


def make_index_dict(label_csv):
    """
    Create a map for the labels from a csv file
    :param label_csv: csv path with labels
    """
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    #del csv_reader
    #gc.collect()
    return index_lookup

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, hop_ms=10):
        """
        Dataset that manages audio recordings
        :param dataset_json_file: json path with all data to load
        :param audio_conf: All the parameters are store in a map -> easier to use
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.hop_ms=hop_ms
        self.data = data_json['data']
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)

        #self.sample_rate=22050
        #self.n_mfcc=40

    def resample_16k(self, data, sr):
        if(sr == 16000): 
            return data, 16000
        elif(sr == 32000):
            return data[:,::2], 16000
        else:
            raise RuntimeError("Unexpected sampling rate %s" % (sr))

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is : Tensor(3, H, W)
        audio is Tensor : (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        waveform, sr = torchaudio.load(datum['wav'])
        waveform, sr = self.resample_16k(waveform, sr)
        waveform = waveform - waveform.mean()

        # To use fbank instead of mfcc
        #fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        #                                          window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.hop_ms)

        mfcc = torchaudio.compliance.kaldi.mfcc(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.hop_ms, num_ceps=128)


        target_length = self.audio_conf.get('target_length')
        n_frames = mfcc.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            mfcc = torch.nn.functional.pad(mfcc, (0, 0, 0, p), mode='constant')
        elif p < 0:
            mfcc = mfcc[0:target_length, :]

        basename = os.path.basename(datum['wav'])

        del waveform, target_length, p, datum, n_frames, sr
        #gc.collect()
        
        # Normalizing the tensor
        mfcc = (mfcc-torch.mean(mfcc))/torch.std(mfcc)

        return mfcc, label_indices, basename

    def __len__(self):
        return len(self.data)