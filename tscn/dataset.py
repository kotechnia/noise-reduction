import os
import torch
import librosa
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torchaudio.transforms import Spectrogram
from pathlib import Path


class TscnDataset(Dataset):
    def __init__(self, sn, sd,
                 n_fft=320, sr=16000, win_len=320,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 save_spectrogram=False, save_pth="/home/jongmin/train/imgs/", remove=False):
        '''

        :param sn: the path for input data
        :param sd: path for ground truth
        :param n_fft: the number of fft points
        :param sr: sampling rate
        :param win_len: window length for hamming window
        '''

        self.data = sn
        self.labels = sd

        self.n_fft = n_fft
        self.sr = sr
        self.win_len = win_len

        self.device = device
        self.save = save_spectrogram
        self.save_pth = save_pth

        Path(self.save_pth).mkdir(parents=True, exist_ok=True)
        if remove:
            os.system(f"rm {os.path.join(save_pth, '*.wav')}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        noisy, _ = librosa.load(self.data[i], sr=self.sr)
        noisy = torch.tensor(noisy)
        clean, _ = librosa.load(self.labels[i], sr=self.sr)
        clean = torch.tensor(clean)

        noisy = Spectrogram(n_fft=self.n_fft, win_length=self.win_len, power=None, return_complex=True)(noisy)
        clean = Spectrogram(n_fft=self.n_fft, win_length=self.win_len, power=None, return_complex=True)(clean)

        if self.save:
            noisy_spec = noisy.log2().abs()
            clean_spec = clean.log2().abs()

            plt.subplot(1, 2, 1)
            plt.title("Clean spectrogram")
            plt.imshow(clean_spec)

            plt.subplot(1, 2, 2)
            plt.title("Noisy spectrogram")
            plt.imshow(noisy_spec)

            plt.savefig(f"{self.data[i].split('.')[-1]}.png")

        return noisy, clean
