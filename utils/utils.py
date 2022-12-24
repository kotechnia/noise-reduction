import librosa
import os
import numpy as np
import subprocess
from pystoi import stoi

import matplotlib.pyplot as plt
import pandas as pd

import re

import soundfile as sf
from datetime import datetime
from dataloader.DataLoader import __CLEAN_COLUMN__, __NOISY_COLUMN__


def file_checker(path_list):
    for path in path_list:
        if os.path.isfile(path) == False:
            print('{} 파일이 존재하지 않습니다.'.format(path))
            


def rnnoise_loss_history(model_path):
    loss_path = os.path.join(model_path, 'training/loss/loss_history.csv')    
    loss_history = pd.read_csv(loss_path)
    plt.figure()
    plt.title('loss_history')
    for column in ['denoise_output_msse', 'val_denoise_output_msse']:
        plt.plot(loss_history[column].to_numpy(), label=column)
    plt.legend()
    plt.show()
    
    
def tscn_loss_history(model_path):
    loss_path = os.path.join(model_path, 'loss_history.csv')
    loss_history = pd.read_csv(loss_path)
    plt.figure()
    for col in loss_history.columns:
        if col == 'train_type':
            continue
        plt.plot(loss_history.loc[:, col], label=col)
    plt.legend()
    plt.show()     
    
    
def calc_estoi(path1, path2):
    sr = 16000
    src1, sr1 = librosa.load(path1, sr=sr)
    src2, sr2 = librosa.load(path2, sr=sr)
    if src1.shape[0] != src2.shape[0]:
        min_len = min(src1.shape[0], src2.shape[0])
        src1 = src1[:min_len]
        src2 = src2[:min_len]
    estoi = stoi(src1, src2, sr, extended=True)
    
    return estoi    
    
def none_tqdm(array):
    return array
    
def denoise_estoi(
    denoise_module,
    dataloader,
    output_dir = 'denoise',
    tqdm=None,
):
    
    oldmask = os.umask(0)
    os.makedirs(output_dir, exist_ok=True, mode=0o0777)
    os.umask(oldmask)
    
    test_info = []
    if tqdm is not None:
        _dataloader = tqdm(dataloader, ascii=True)
    else:
        _dataloader = dataloader
    
    for clean_path, noisy_path in _dataloader:

        noisy_name = noisy_path.split('/')[-1]
        denoise_path = os.path.join(output_dir, noisy_name)
        denoise_path = os.path.abspath(denoise_path)

        denoise_module(
            src_path = noisy_path,
            dst_path = denoise_path,
        )
        
        estoi = calc_estoi(
            clean_path, 
            denoise_path
        )

        data = {__CLEAN_COLUMN__:clean_path,
                __NOISY_COLUMN__:noisy_path,
                'denoise_path':denoise_path,
                'estoi':estoi}
        test_info.append(data)
    
    df = pd.DataFrame(test_info)
    df.to_csv(os.path.join(output_dir, 'info.csv'), mode='w', header=True, index=False, encoding='utf-8-sig')            
    return df

def trans_path(src_path, trans = {'./':'./'}):
    
    for key, value in trans.items():
        dst_path = re.sub(key, value, src_path)
        return dst_path
    
    return False

def make_directory(path):
    os.umask(0)
    os.makedirs('/mnt/hdd4t/tt', mode=0o0777, exist_ok=True)
    
    
class DivideWAVFile():
    def get_file_name(self, path, index):        
        index=index+1
        directory_path, file_name = os.path.split(path)
        file_name, _ = os.path.splitext(file_name)        
        file_name += f'_{index:03}.wav'
        
        if self.output_dir is None:
            file_path = os.path.join(directory_path, file_name)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            file_path = os.path.join(self.output_dir, file_name)
        
        if self.abspath:
            file_path = os.path.abspath(file_path)
        
        return file_path
    
    def wav_file_save(self, path, data, sr):
        sf.write(path, data, sr, format='WAV', endian='LITTLE', subtype='PCM_16')
        return
    
    
    def __init__(self, df=None, output_dir=None, abspath=False, ignore_remainder = 0.0, sr=16000, sec=30., csv_file_path = None, tqdm=None):
        if df is None:
            self.df = pd.read_excel('excel/diff_insight_data_modiy.xlsx')
        else :
            self.df = df
            
        self.sr = sr
        self.sec = sec
        
        self.output_dir = output_dir
        self.abspath = abspath
        
        self.ignore_remainder = ignore_remainder
        self.tqdm = tqdm
        
        if csv_file_path is None:
            self.csv_file_path = datetime.now().strftime("%Y%m%d_%H%M%S.csv")
        else:
            self.csv_file_path = csv_file_path
        
        #print(self.csv_file_path)
            
        
    def __iter__(self):
        for i in self.tqdm(range(len(self.df)), ascii=True):
            sd_file_path = self.df.loc[i, __CLEAN_COLUMN__]
            sn_file_path = self.df.loc[i, __NOISY_COLUMN__]
            
            sd_signal, sd_sr = librosa.load(sd_file_path, sr=self.sr)
            sn_signal, sn_sr = librosa.load(sn_file_path, sr=self.sr)
            
            if sd_sr != sn_sr and self.sr == None:
                continue
                
            if sd_signal.shape[0] != sn_signal.shape[0]:
                continue
                
            
            sr = sd_sr
            stride = int(sr * self.sec)
                
            total = sd_signal.shape[0]
            iteration = total // stride
            
            for idx in range(iteration):
                
                _sd_file_path = self.get_file_name(sd_file_path, idx)
                _sn_file_path = self.get_file_name(sn_file_path, idx)
                
                _sd_signal = sd_signal[(idx * stride): ((idx+1) * stride)]
                _sn_signal = sn_signal[(idx * stride): ((idx+1) * stride)]
                
                
                self.wav_file_save(_sd_file_path, _sd_signal, self.sr)
                self.wav_file_save(_sn_file_path, _sn_signal, self.sr)
                
                if os.path.isfile(self.csv_file_path):
                    df_temp = pd.DataFrame(zip([_sd_file_path], [_sn_file_path]), columns=[__CLEAN_COLUMN__, __NOISY_COLUMN__])
                    df_temp.to_csv(self.csv_file_path, mode='a', header=False, index=False)            
                else:
                    df_temp = pd.DataFrame(zip([_sd_file_path], [_sn_file_path]), columns=[__CLEAN_COLUMN__, __NOISY_COLUMN__])
                    df_temp.to_csv(self.csv_file_path, mode='w', header=True, index=False)            
                
                #print(f'{sd_file_path}')
                #print(f'-> {_sd_file_path}')
                #print(f'{sn_file_path}')
                #print(f'-> {_sn_file_path}')
                
                yield _sd_file_path, _sn_file_path
                
            _sd_signal = sd_signal[(iteration * stride):]
            _sn_signal = sn_signal[(iteration * stride):]
                
            if _sd_signal.shape[0] <= self.ignore_remainder * stride:
                continue
            
            _sd_file_path = self.get_file_name(sd_file_path, iteration)
            _sn_file_path = self.get_file_name(sn_file_path, iteration)
            
            padding_size = (iteration + 1) * stride - total
            padding = np.zeros([padding_size])
            
            _sd_signal = np.concatenate([_sd_signal, padding])
            _sn_signal = np.concatenate([_sn_signal, padding])
            
            self.wav_file_save(_sd_file_path, _sd_signal, self.sr)
            self.wav_file_save(_sn_file_path, _sn_signal, self.sr)
            
            if os.path.isfile(self.csv_file_path):
                df_temp = pd.DataFrame(zip([_sd_file_path], [_sn_file_path]), columns=[__CLEAN_COLUMN__, __NOISY_COLUMN__])
                df_temp.to_csv(self.csv_file_path, mode='a', header=False, index=False)            
            else:
                df_temp = pd.DataFrame(zip([_sd_file_path], [_sn_file_path]), columns=[__CLEAN_COLUMN__, __NOISY_COLUMN__])
                df_temp.to_csv(self.csv_file_path, mode='w', header=True, index=False)            
                
            #print(f'{sd_file_path}')
            #print(f'-> {_sd_file_path}')
            #print(f'{sn_file_path}')
            #print(f'-> {_sn_file_path}')
                
            yield _sd_file_path, _sn_file_path
    
