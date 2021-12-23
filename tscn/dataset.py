import glob
import os
import torch
import librosa
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from dataloader.DataLoader import DataLoader as DB
from torchaudio.transforms import Spectrogram
from pathlib import Path

def GetKeyFromPath(path):
    directory_path, file_name, = os.path.split(path)
    key, _ = file_name.split('.')
    return key

class TscnDataset(Dataset):
    def __init__(self, data_loader,
                 n_fft=320, sr=16000, win_len=320,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 #save_spectrogram=False,
                 save_preprocess_path=None,
                 show_preprocess = False,
                ):
        '''

        :param sn: the path for input data
        :param sd: path for ground truth
        :param n_fft: the number of fft points
        :param sr: sampling rate
        :param win_len: window length for hamming window
        '''

        self.data = [sn for _, sn in data_loader]
        self.labels = [sd for sd, _ in data_loader]

        self.n_fft = n_fft
        self.sr = sr
        self.win_len = win_len

        self.device = device
        
        self.save_pth = save_preprocess_path
        self.show_preprocess = show_preprocess
        if self.save_pth is not None:
            
            self.flags = [True for _, _ in data_loader]
            
            self.save = True
            Path(self.save_pth).mkdir(parents=True, exist_ok=True)
        else:
            self.save = False
            

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
            if self.flags[i]:
                # 2021-11-04 jaewon mod
                noisy_spec = noisy.log2().abs()
                clean_spec = clean.log2().abs()
                
                sd_file_name = GetKeyFromPath(self.data[i])
                sn_file_name = GetKeyFromPath(self.labels[i])
                
                plt.figure(figsize=(8,9))
                plt.subplot(2, 1, 1)
                #plt.title("Clean spectrogram")
                plt.title("{} spectrogram".format(sd_file_name))
                plt.imshow(clean_spec, aspect=8, vmin=0, vmax=8)

                plt.subplot(2, 1, 2)
                #plt.title("Noisy spectrogram")
                plt.title("{} spectrogram".format(sn_file_name))
                plt.imshow(noisy_spec, aspect=8, vmin=0, vmax=8)
                
                save_path = os.path.join(self.save_pth, "{}.png".format(sd_file_name))
                #print(save_path)
                plt.savefig(save_path)
                
                if self.show_preprocess:
                    plt.show()
                
                self.flags[i] = False

        return noisy, clean


def TscnLoader(
            df = None,
            path = None,
            batch_size=6,
            save_preprocess_path=None,
            show_preprocess=False,
    ):
        data = DB(
            df=df,
            path=path
        )

        dataset = TscnDataset(data_loader=data, save_preprocess_path=save_preprocess_path, show_preprocess=show_preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        #loader = DataLoader(dataset, batch_size=batch_size)

        return loader
