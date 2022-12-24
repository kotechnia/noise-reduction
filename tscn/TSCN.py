import torch
import torch.nn as nn
import tqdm
import os
import librosa
import pandas as pd
import numpy as np
import glob

from tscn.CME import CMENet
from tscn.CSR import CSRNet
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchaudio.transforms import Spectrogram

from copy import deepcopy
from pathlib import Path
from dataloader.DataLoader import DataLoader as DB
from tscn.dataset import TscnDataset
#from scipy.io.wavfile import write
from pystoi import stoi
import soundfile as sf
from datetime import datetime

def current_time():
	return datetime.now().strftime("%Y%m%d-%H%M%S")

        
def normalize(audio, target_level=-25):
    EPS = np.finfo(float).eps
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio, scalar

class TSCN_Module(nn.Module):
    def __init__(self, device, n_cme_blocks=3, n_csr_blocks=2, multi=False):
        """

        :param device: The device where to train the network
        :param n_cme_blocks: number of CME blocks
        :param n_csr_blocks: number of CSR blocks
        :param multi: decides whether to train on multi GPU or single
        """
        super(TSCN_Module, self).__init__()

        if multi:  # for multi gpu training
            self.cme = nn.DataParallel(CMENet(n_blocks=n_cme_blocks))
            self.csr = nn.DataParallel(CSRNet(n_blocks=n_csr_blocks))
        else:
            self.cme = CMENet(n_blocks=n_cme_blocks)
            self.csr = CSRNet(n_blocks=n_csr_blocks)

        self.device = device

    def forward(self, x):
        if x.dtype == torch.float32:
            x = torch.complex(x, x)

        mag, phase = x.abs().type(torch.FloatTensor).to(self.device), x.angle().type(torch.FloatTensor).to(self.device)

        cme_mag = self.cme(mag)
        CCS = torch.complex(cme_mag * torch.cos(phase).to(self.device), cme_mag * torch.sin(phase).to(self.device))

        CCS = torch.view_as_real(CCS).permute(0, 3, 1, 2).to(self.device)

        x = torch.view_as_real(x).permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)

        output = self.csr(CCS, x)
        output = output + CCS

        output = output.permute(0, 2, 3, 1).contiguous().to(self.device)

        output = torch.view_as_complex(output).to(self.device)

        return cme_mag, output


class TSCN:
    def __init__(
        self,
        weight_pth="/home/jongmin/train/weights",
        transfer=False,
        device="cuda",
        multi=False,
        cme_filename="TSCN_CME.pth",
        csr_filename="TSCN_CSR.pth",        
        n_fft=320,
        win_len=320,
#         loss_weight_coefficient=0.1,
#         cme_epochs=400, cme_lr=0.001,
#         finetuning_epochs=40, finetuning_lr=0.0001,
#         csr_epochs=400, csr_lr=0.0001,
        #batch_size=6,
        #model_select=None,
        #cutoff=False,
        #train=True,
        #val=True,
        # sec=30,
        # sr=16000,
        # all_data=True,
        # db_update=False,
        # database_path="",
    ):

        # static value
        self.sr = 16000  # sampling rate
        self.sec = 30  # the length of training data in seconds
        self.cutoff = False  # if true, cuts off the remainder which is smaller than sec
        
        self.n_fft = n_fft  # points for FFT
        self.win_len = win_len  # window length of Hamming window
        
        # parameter setup
        self.tscn = TSCN_Module(device=device, multi=multi).to(device)
        self.device = device
        self.multi = multi
        self.transfer = transfer  # if true, loads the pretrained model weights

        # path settings
        self.weight_pth = weight_pth  # path to weight file
        self.cme_filename = cme_filename  # filename for CME weights
        self.csr_filename = csr_filename  # filename for CSR weights

        self.cme_file_path = os.path.join(self.weight_pth, self.cme_filename)
        self.csr_file_path = os.path.join(self.weight_pth, self.csr_filename)
        
        #self.cme_loss_path = os.path.join(self.weight_pth, 'cme_loss.csv')
        #self.finetuning_loss_path = os.path.join(self.weight_pth, 'finetuning_loss.csv')
        #self.csr_loss_path = os.path.join(self.weight_pth, 'csr_loss.csv')
        
        self.cme_loss_path = os.path.join(self.weight_pth, 'loss_history.csv')
        self.finetuning_loss_path = os.path.join(self.weight_pth, 'loss_history.csv')
        self.csr_loss_path = os.path.join(self.weight_pth, 'loss_history.csv')
        
        # make sure that the paths exists
        Path(self.weight_pth).mkdir(parents=True, exist_ok=True)
        
        # load pretrained weights
        if self.transfer:
            if self.multi:
                self.tscn.cme.module.load_state_dict(torch.load(self.cme_file_path))
                self.tscn.csr.module.load_state_dict(torch.load(self.csr_file_path))
            else:
                self.tscn.cme.load_state_dict(torch.load(self.cme_file_path))
                self.tscn.csr.load_state_dict(torch.load(self.csr_file_path))



    def fit(self, 
            model_select=None,
            train_loader=None,
            val_loader = None,
            loss_weight_coefficient=0.1,
            cme_epochs=None,
            cme_lr=0.001,
            finetuning_epochs=None,
            finetuning_lr=0.0001,
            csr_epochs=None,
            csr_lr=0.0001,
           ):
        
        # selects which model to train
        # options = "cme", "finetune", "csr", None
        # if None, trains the entire model
        self.model_select = model_select

        # epochs
        self.cme_epochs = cme_epochs
        self.finetuning_epochs = finetuning_epochs
        self.csr_epochs = csr_epochs

        #self.batch_size = batch_size

        self.loss_weight_coefficient = loss_weight_coefficient  # gives penalty to the CME loss

        # learning rates
        self.cme_lr = cme_lr
        self.finetuning_lr = finetuning_lr
        self.csr_lr = csr_lr

        # optimizers
        self.cme_optim = Adam(lr=cme_lr, params=self.tscn.cme.parameters())
        self.finetune_optim = Adam(lr=finetuning_lr, params=self.tscn.csr.parameters())
        self.csr_optim = Adam(params=self.tscn.parameters(), lr=csr_lr)
        
        #if self.train:
        if train_loader is not None:
            
            if self.cme_epochs is not None:
                print("################ training CME ... ################")
                self.cme_epochs = int(self.cme_epochs)
                # train CME
                self.train_CME(loader=train_loader, val_loader=val_loader)
                if self.multi:
                    self.tscn.cme.module.load_state_dict(torch.load(self.cme_file_path))
                else:
                    self.tscn.cme.load_state_dict(torch.load(self.cme_file_path))

            if self.finetuning_epochs is not None:
                print("################ finetuning CSR ... ################")
                self.tscn.cme.eval()
                self.finetuning_epochs = int(self.finetuning_epochs)

                # fine tune CSR
                self.finetune_CSR(loader=train_loader, val_loader=val_loader)

                if self.multi:
                    self.tscn.csr.module.load_state_dict(torch.load(self.csr_file_path))
                else:
                    self.tscn.csr.load_state_dict(torch.load(self.csr_file_path))

            if self.csr_epochs is not None:
                print("################ training CSR ... ################")
                self.tscn.train()
                self.csr_epochs = int(csr_epochs)

                # train CSR
                self.train_CSR(loader=train_loader, val_loader=val_loader)

                if self.multi:
                    self.tscn.cme.module.load_state_dict(torch.load(self.cme_file_path))
                    self.tscn.csr.module.load_state_dict(torch.load(self.csr_file_path))
                else:
                    self.tscn.cme.load_state_dict(torch.load(self.cme_file_path))
                    self.tscn.csr.load_state_dict(torch.load(self.csr_file_path))
                    

    def train_CME(self, loader, val_loader=None):
        # best_loss is a random number
        # The smallest loss is the best loss
        best_loss = 100

        train_log_cme = []
        train_log_csr = []
        val_log_cme = []        
        val_log_csr = [] 
        train_type = []
        
        for epoch in range(self.cme_epochs):
            self.tscn.cme.train()
            iterator = tqdm.tqdm(loader, ascii=True)
            avg_loss = 0
            for data, label in iterator:
                self.cme_optim.zero_grad()
                
                data_m = data.abs().type(torch.FloatTensor).to(self.device)
                data_p = data.angle().type(torch.FloatTensor).to(self.device)
                label_m = label.abs().type(torch.FloatTensor).to(self.device)
                label_p = label.angle().type(torch.FloatTensor).to(self.device)

                output = self.tscn.cme(data_m)

                loss = nn.MSELoss()(output, label_m)
                avg_loss += loss.item() / len(loader)
                iterator.set_description(f"{current_time()} epoch:{epoch:3} - cme_loss:{avg_loss:.4f}, batch_loss={loss.item():.4f}")
                loss.backward()
                self.cme_optim.step()
                
            train_log_cme.append(avg_loss)
            train_log_csr.append(None)
            train_type.append('cme')
            
            if val_loader is not None:
                with torch.no_grad():
                    self.tscn.cme.eval()
                    iterator = tqdm.tqdm(val_loader, ascii=True)
                    val_loss = 0

                    for data, label in iterator:
                        data_m = data.abs().type(torch.FloatTensor).to(self.device)
                        data_p = data.angle().type(torch.FloatTensor).to(self.device)
                        label_m = label.abs().type(torch.FloatTensor).to(self.device)
                        label_p = label.angle().type(torch.FloatTensor).to(self.device)

                        output = self.tscn.cme(data_m)

                        loss = nn.MSELoss()(output, label_m)
                        val_loss += loss.item() / len(val_loader)
                        iterator.set_description(
                            f"{current_time()} epoch:{epoch:3} - valid cme_loss:{val_loss:.4f}, batch_loss={loss.item():.4f}")
                        
                val_log_cme.append(val_loss)        
                val_log_csr.append(None)
            else:
                val_log_cme.append(None)        
                val_log_csr.append(None)

            if avg_loss <= best_loss:
                if self.multi:
                    torch.save(self.tscn.cme.module.state_dict(), self.cme_file_path)
                else:
                    torch.save(self.tscn.cme.state_dict(), self.cme_file_path)
                best_loss = avg_loss

                if self.transfer:
                    if os.path.isfile(self.cme_loss_path):
                        pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                     columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss', 'train_type']).to_csv(self.cme_loss_path, mode='a', header=False, index=False)            
                    else:
                        pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                     columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss', 'train_type']).to_csv(self.cme_loss_path, mode='w', header=True, index=False)                        
                else:
                    pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                     columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss','train_type']).to_csv(self.cme_loss_path, mode='w', header=True, index=False)            
                    
                self.transfer = True
                train_log_cme = []
                train_log_csr = []
                val_log_cme = []        
                val_log_csr = [] 
                train_type = []
        
        return 

    def finetune_CSR(self, loader, val_loader=None):
        # best_loss is a random number
        # The smallest loss is the best loss
        best_loss = 100

        train_log_csr = []
        train_log_cme = []
        val_log_csr = []
        val_log_cme = []
        train_type = []
        
        self.tscn.cme.eval()

        for epoch in range(self.finetuning_epochs):
            self.tscn.csr.train()
            iterator = tqdm.tqdm(loader, ascii=True)
            avg_loss = 0
            cme_avg = 0
            for data, label in iterator:
                self.finetune_optim.zero_grad()
                label_m = label.abs().type(torch.FloatTensor).to(self.device)
                label_p = label.angle().type(torch.FloatTensor).to(self.device)

                label_r = label.real.type(torch.FloatTensor).to(self.device)
                label_i = label.imag.type(torch.FloatTensor).to(self.device)

                cme_mag, output = self.tscn(data)
                real, imag = output.real, output.imag
                mag, phase = output.abs(), output.angle()

                cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag, label_i) + cme_loss

                avg_loss += loss.item() / len(loader)
                cme_avg += cme_loss.item() * 10 / len(loader)
                iterator.set_description(f"{current_time()} epoch:{epoch:3} - csr_loss:{avg_loss:.4f}, cme_loss={cme_avg:.4f}")
                loss.backward()

                self.finetune_optim.step()
                
            train_log_csr.append(avg_loss)
            train_log_cme.append(cme_avg)
            train_type.append('finetune')

            if val_loader is not None:
                with torch.no_grad():
                    self.tscn.csr.eval()

                    iterator = tqdm.tqdm(val_loader, ascii=True)
                    val_loss = 0
                    val_cme_loss = 0
                    for data, label in iterator:
                        label_m = label.abs().type(torch.FloatTensor).to(self.device)
                        label_p = label.angle().type(torch.FloatTensor).to(self.device)

                        label_r = label.real.type(torch.FloatTensor).to(self.device)
                        label_i = label.imag.type(torch.FloatTensor).to(self.device)

                        cme_mag, output = self.tscn(data)
                        real, imag = output.real, output.imag
                        mag, phase = output.abs(), output.angle()

                        cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                        loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag,
                                                                                                       label_i) + cme_loss
                        val_loss += loss.item() / len(val_loader)
                        val_cme_loss += cme_loss.item() * 10 / len(val_loader)
                        #iterator.set_description(f"epoch:{epoch:3} - validation loss:{val_loss:.4f}, batch_loss={loss.item():.4f}")
                        iterator.set_description(f"{current_time()} epoch:{epoch:3} - valid csr_loss:{val_loss:.4f}, cme_loss={val_cme_loss:.4f}")

                val_log_csr.append(val_loss)
                val_log_cme.append(val_cme_loss)
            else:
                val_log_csr.append(None)
                val_log_cme.append(None)
                        
            if avg_loss <= best_loss:
                if self.multi:
                    torch.save(self.tscn.csr.module.state_dict(), self.csr_file_path)
                else:
                    torch.save(self.tscn.csr.state_dict(), self.csr_file_path)
                best_loss = avg_loss
                
                if os.path.isfile(self.finetuning_loss_path):
                    pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                 columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss', 'train_type']).to_csv(self.finetuning_loss_path, mode='a', header=False, index=False)
                else:
                    pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                 columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss', 'train_type']).to_csv(self.finetuning_loss_path, mode='w', header=True, index=False)            
                train_log_csr = []
                train_log_cme = []
                val_log_csr = []
                val_log_cme = []
                train_type = []
                
        return 

    def train_CSR(self, loader, val_loader=None):
        # best_loss is a random number
        # The smallest loss is the best loss
        best_loss = 100

        train_log_csr = []
        train_log_cme = []
        val_log_csr = []
        val_log_cme = []
        train_type = []
        
        for epoch in range(self.csr_epochs):
            self.tscn.train()
            iterator = tqdm.tqdm(loader, ascii=True)
            avg_loss = 0
            cme_avg = 0
            for data, label in iterator:
                self.csr_optim.zero_grad()
                label_m = label.abs().type(torch.FloatTensor).to(self.device)
                label_p = label.angle().type(torch.FloatTensor).to(self.device)

                label_r = label.real.type(torch.FloatTensor).to(self.device)
                label_i = label.imag.type(torch.FloatTensor).to(self.device)

                cme_mag, output = self.tscn(data)
                real, imag = output.real, output.imag
                mag, phase = output.abs(), output.angle()

                cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag, label_i) + cme_loss

                avg_loss += loss.item() / len(loader)
                cme_avg += cme_loss.item() * 10 / len(loader)
                iterator.set_description(f"{current_time()} epoch:{epoch:3} - csr_loss:{avg_loss:.4f}, cme_loss={cme_avg:.4f}")
                loss.backward()

                self.csr_optim.step()
                
            train_log_csr.append(avg_loss)
            train_log_cme.append(cme_avg)
            train_type.append('csr')

            if val_loader is not None:
                with torch.no_grad():
                    self.tscn.eval()

                    iterator = tqdm.tqdm(val_loader, ascii=True)
                    val_loss = 0
                    val_cme_loss = 0
                    
                    for data, label in iterator:
                        label_m = label.abs().type(torch.FloatTensor).to(self.device)
                        label_p = label.angle().type(torch.FloatTensor).to(self.device)

                        label_r = label.real.type(torch.FloatTensor).to(self.device)
                        label_i = label.imag.type(torch.FloatTensor).to(self.device)

                        cme_mag, output = self.tscn(data)
                        real, imag = output.real, output.imag
                        mag, phase = output.abs(), output.angle()

                        cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                        loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag,
                                                                                                       label_i) + cme_loss
                        val_loss += loss.item() / len(val_loader)
                        val_cme_loss += cme_loss.item() * 10 / len(val_loader)
                        
                        #iterator.set_description(f"epoch:{epoch:3} \t validation loss:{val_loss:.4f}, batch_loss={loss.item():.4f}")
                        iterator.set_description(f"{current_time()} epoch:{epoch:3} - valid csr_loss:{val_loss:.4f}, cme_loss={val_cme_loss:.4f}")
                        
                val_log_csr.append(val_loss) 
                val_log_cme.append(val_cme_loss)
            else:
                val_log_csr.append(None) 
                val_log_cme.append(None)

            if avg_loss <= best_loss:
                if self.multi:
                    torch.save(self.tscn.cme.module.state_dict(), self.cme_file_path)
                    torch.save(self.tscn.csr.module.state_dict(), self.csr_file_path)
                else:
                    torch.save(self.tscn.cme.state_dict(), self.cme_file_path)
                    torch.save(self.tscn.csr.state_dict(), self.csr_file_path)
                best_loss = avg_loss

                if os.path.isfile(self.csr_loss_path):
                    pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                 columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss', 'train_type']).to_csv(self.csr_loss_path, mode='a', header=False, index=False)            
                else:
                    pd.DataFrame(zip(train_log_csr, train_log_cme, val_log_csr, val_log_cme, train_type),
                                 columns=['csr_loss', 'cme_loss', 'val_csr_loss', 'val_cme_loss', 'train_type']).to_csv(self.csr_loss_path, mode='w', header=True, index=False)                        
                train_log_csr = []
                train_log_cme = []
                val_log_csr = []
                val_log_cme = []
                train_type = []
        return 

    def inference(self, src_path, dst_path):
        def split_infer(filename, sec, sr=16000, cutoff=True):
            data, sr = librosa.load(filename, sr=sr)
            
            batches = []
            
            _len = data.shape[0]

            i = 0
            padded = 0
            batch_size = sr * sec
            while i < len(data):
                if i + batch_size <= len(data):
                    batches.append(data[i:i + batch_size])
                else:
                    padding_size = i + batch_size - _len
                    batch = np.concatenate([data[i:len(data)], np.zeros([padding_size])])
                    batches.append(batch)
                    
                i = i + batch_size

            return _len, batches

        #model = TSCN_Module(device=self.device).to(self.device)
        
        if self.multi:
            self.tscn.cme.module.load_state_dict(torch.load(self.cme_file_path, map_location=self.device))
            self.tscn.csr.module.load_state_dict(torch.load(self.csr_file_path, map_location=self.device))
        else:
            self.tscn.cme.load_state_dict(torch.load(self.cme_file_path, map_location=self.device))
            self.tscn.csr.load_state_dict(torch.load(self.csr_file_path, map_location=self.device))
            
        self.tscn.eval()

        infer_signal = []
        src_len, noisy_batches = split_infer(src_path.strip(), sr=self.sr, sec=self.sec, cutoff=self.cutoff)
        with torch.no_grad():
            for batch in noisy_batches:
                if len(batch) == 0:
                    continue
                signal = torch.tensor(batch)
                signal = Spectrogram(n_fft=self.n_fft, win_length=self.win_len, power=None, return_complex=True)(signal)
                signal = torch.unsqueeze(signal, dim=0).to(self.device)

                cme_mag, output = self.tscn(signal)

                wave = torch.istft(output, n_fft=self.n_fft, hop_length=int(self.win_len/2), win_length=self.win_len, length=len(batch))
                wave = torch.flatten(wave).cpu()
                wave = wave.detach().numpy()

                infer = wave.tolist()
                infer_signal.extend(infer)
                
        infer_signal = np.array(infer_signal[:src_len]).astype(np.float32)
        infer_signal, _ = normalize(infer_signal)

        sf.write(dst_path, infer_signal, self.sr, format='WAV', endian='LITTLE', subtype='PCM_16')        
        #sf.write(dst_path, np.array(infer_signal[:src_len]/scalar).astype(np.float32), self.sr, format='WAV', endian='LITTLE', subtype='PCM_16')        
