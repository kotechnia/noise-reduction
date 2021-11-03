import torch
import torch.nn as nn
import tqdm
import os
import librosa
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
from scipy.io.wavfile import write
from pystoi import stoi


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
            self, weight_pth="/home/jongmin/train/weights",
            transfer=False,
            train=True, val=True,
            model_select=None,
            sr=16000, batch_size=6,
            cme_epochs=400, cme_lr=0.001,
            finetuning_epochs=40, finetuning_lr=0.0001,
            csr_epochs=400, csr_lr=0.0001,
            device="cuda",
            n_fft=320, win_len=320, loss_weight_coefficient=0.1, multi=False,
            sec=30, cutoff=False,
            all_data=True, db_update=False, database_path="",
            cme_filename="TSCN_CME.pth", csr_filename="TSCN_CSR.pth"
                 ):

        # parameter setup
        self.tscn = TSCN_Module(device=device, multi=multi).to(device)
        self.device = device
        self.multi = multi
        self.transfer = transfer  # if true, loads the pretrained model weights

        # path settings
        self.weight_pth = weight_pth  # path to weight file
        self.cme_filename = cme_filename  # filename for CME weights
        self.csr_filename = csr_filename  # filename for CSR weights

        # make sure that the paths exists
        Path(self.weight_pth).mkdir(parents=True, exist_ok=True)

        # load pretrained weights
        if self.transfer:
            if self.multi:
                self.tscn.cme.module.load_state_dict(torch.load(os.path.join(self.weight_pth, self.cme_filename)))
                self.tscn.csr.module.load_state_dict(torch.load(os.path.join(self.weight_pth, self.csr_filename)))
            else:
                self.tscn.cme.load_state_dict(torch.load(os.path.join(self.weight_pth, self.cme_filename)))
                self.tscn.csr.load_state_dict(torch.load(os.path.join(self.weight_pth, self.csr_filename)))

        self.train = train  # if true, trains the network
        self.val = val  # if true, uses validation data when training

        # selects which model to train
        # options = "cme", "finetune", "csr", None
        # if None, trains the entire model
        self.model_select = model_select

        self.sr = sr  # sampling rate
        self.sec = sec  # the length of training data in seconds
        self.cutoff = cutoff  # if true, cuts off the remainder which is smaller than sec

        # epochs
        self.cme_epochs = cme_epochs
        self.finetuning_epochs = finetuning_epochs
        self.csr_epochs = csr_epochs

        self.batch_size = batch_size

        self.loss_weight_coefficient = loss_weight_coefficient  # gives penalty to the CME loss

        # learning rates
        self.cme_lr = cme_lr
        self.finetuning_lr = finetuning_lr
        self.csr_lr = csr_lr

        # optimizers
        self.cme_optim = Adam(lr=cme_lr, params=self.tscn.cme.parameters())
        self.finetune_optim = Adam(lr=finetuning_lr, params=self.tscn.csr.parameters())
        self.csr_optim = Adam(params=self.tscn.parameters(), lr=csr_lr)

        self.all_data = all_data  # if true, use all data from DB
        self.db_update = db_update  # if true, updates DB's training flag
        self.database_path = database_path  # path to DB setting file

        self.n_fft = n_fft  # points for FFT
        self.win_len = win_len  # window length of Hamming window

    def fit(self, train_loader, val_loader):

        if self.train:
            if self.model_select == None or self.model_select == "cme":
                print("################ training CME ... ################")

                # train CME
                state_dict = self.train_CME(loader=train_loader, val_loader=val_loader)
                if self.multi:
                    self.tscn.cme.module.load_state_dict(state_dict)
                else:
                    self.tscn.cme.load_state_dict(state_dict)

            if self.model_select == None or self.model_select == "csr" or self.model_select == "finetune":
                print("################ finetuning CSR ... ################")
                self.tscn.cme.eval()

                # fine tune CSR
                state_dict = self.finetune_CSR(loader=train_loader, val_loader=val_loader)

                if self.multi:
                    self.tscn.csr.module.load_state_dict(state_dict)
                else:
                    self.tscn.csr.load_state_dict(state_dict)

            if self.model_select == None or self.model_select == "csr":
                print("################ training CSR ... ################")
                self.tscn.train()

                # train CSR
                cme, csr = self.train_CSR(loader=train_loader, val_loader=val_loader)

                if self.multi:
                    self.tscn.cme.module.load_state_dict(cme)
                    self.tscn.csr.module.load_state_dict(csr)
                else:
                    self.tscn.cme.load_state_dict(cme)
                    self.tscn.csr.load_state_dict(csr)

    def train_CME(self, loader, val_loader=None):
        # best_loss is a random number
        # The smallest loss is the best loss
        best_loss = 100
        best = self.tscn.state_dict()

        for epoch in range(self.cme_epochs):
            self.tscn.cme.train()
            iterator = tqdm.tqdm(loader)
            avg_loss = 0
            for data, label in iterator:
                self.cme_optim.zero_grad()
                data_m, data_p = data.abs().type(torch.FloatTensor).to(self.device), data.angle().type(torch.FloatTensor).to(
                    self.device)
                label_m, label_p = label.abs().type(torch.FloatTensor).to(self.device), label.angle().type(
                    torch.FloatTensor).to(
                    self.device)

                output = self.tscn.cme(data_m)

                loss = nn.MSELoss()(output, label_m)
                avg_loss += loss.item() / len(loader)
                iterator.set_description(f"epoch:{epoch} \t loss:{avg_loss:.4f}, batch_loss={loss.item():.4f}")
                loss.backward()
                self.cme_optim.step()

            if val_loader is not None:
                with torch.no_grad():
                    self.tscn.cme.eval()
                    iterator = tqdm.tqdm(val_loader)
                    val_loss = 0

                    for data, label in iterator:
                        data_m, data_p = data.abs().type(torch.FloatTensor).to(self.device), data.angle().type(
                            torch.FloatTensor).to(
                            self.device)
                        label_m, label_p = label.abs().type(torch.FloatTensor).to(self.device), label.angle().type(
                            torch.FloatTensor).to(
                            self.device)

                        output = self.tscn.cme(data_m)

                        loss = nn.MSELoss()(output, label_m)
                        val_loss += loss.item() / len(val_loader)
                        iterator.set_description(
                            f"epoch:{epoch} \t validation loss:{val_loss:.4f}, batch_loss={loss.item():.4f}")

            if avg_loss <= best_loss:
                if self.multi:
                    best = deepcopy(self.tscn.cme.module.state_dict())
                else:
                    best = deepcopy(self.tscn.cme.state_dict())
                best_loss = avg_loss

        torch.save(best, os.path.join(self.weight_pth, self.cme_filename))
        return best

    def finetune_CSR(self, loader, val_loader=None):
        # best_loss is a random number
        # The smallest loss is the best loss
        best_loss = 100
        best = deepcopy(self.tscn.csr.state_dict())

        self.tscn.cme.eval()

        for epoch in range(self.finetuning_epochs):
            self.tscn.csr.train()
            iterator = tqdm.tqdm(loader)
            avg_loss = 0
            cme_avg = 0
            for data, label in iterator:
                self.finetune_optim.zero_grad()
                label_m, label_p = label.abs().type(torch.FloatTensor).to(self.device), label.angle().type(torch.FloatTensor).to(
                    self.device)

                label_r = label.real.type(torch.FloatTensor).to(self.device)
                label_i = label.imag.type(torch.FloatTensor).to(self.device)

                cme_mag, output = self.tscn(data)
                real, imag = output.real, output.imag
                mag, phase = output.abs(), output.angle()

                cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag, label_i) + cme_loss

                avg_loss += loss.item() / len(loader)
                cme_avg += cme_loss.item() * 10 / len(loader)
                iterator.set_description(f"epoch:{epoch} \t csr loss:{avg_loss:.4f}, cme_loss={cme_avg:.4f}")
                loss.backward()

                self.finetune_optim.step()

            if val_loader is not None:
                with torch.no_grad():
                    self.tscn.csr.eval()

                    iterator = tqdm.tqdm(val_loader)
                    val_loss = 0
                    for data, label in iterator:
                        label_m, label_p = label.abs().type(torch.FloatTensor).to(self.device), label.angle().type(
                            torch.FloatTensor).to(
                            self.device)

                        label_r = label.real.type(torch.FloatTensor).to(self.device)
                        label_i = label.imag.type(torch.FloatTensor).to(self.device)

                        cme_mag, output = self.tscn(data)
                        real, imag = output.real, output.imag
                        mag, phase = output.abs(), output.angle()

                        cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                        loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag,
                                                                                                       label_i) + cme_loss
                        val_loss += loss.item() / len(val_loader)
                        iterator.set_description(
                            f"epoch:{epoch} \t validation loss:{val_loss:.4f}, batch_loss={loss.item():.4f}")

            if avg_loss <= best_loss:
                if self.multi:
                    best = deepcopy(self.tscn.csr.module.state_dict())
                else:
                    best = deepcopy(self.tscn.csr.state_dict())
                best_loss = avg_loss

        torch.save(best, os.path.join(self.weight_pth, self.csr_filename))

        return best

    def train_CSR(self, loader, val_loader=None):
        # best_loss is a random number
        # The smallest loss is the best loss
        best_loss = 100
        best_csr = deepcopy(self.tscn.csr.state_dict())
        best_cme = deepcopy(self.tscn.cme.state_dict())

        for epoch in range(self.csr_epochs):
            self.tscn.csr.train()
            iterator = tqdm.tqdm(loader)
            avg_loss = 0
            cme_avg = 0
            for data, label in iterator:
                self.csr_optim.zero_grad()
                label_m, label_p = label.abs().type(torch.FloatTensor).to(self.device), label.angle().type(torch.FloatTensor).to(
                    self.device)

                label_r = label.real.type(torch.FloatTensor).to(self.device)
                label_i = label.imag.type(torch.FloatTensor).to(self.device)

                cme_mag, output = self.tscn(data)
                real, imag = output.real, output.imag
                mag, phase = output.abs(), output.angle()

                cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag, label_i) + cme_loss

                avg_loss += loss.item() / len(loader)
                cme_avg += cme_loss.item() * 10 / len(loader)
                iterator.set_description(f"epoch:{epoch} \t csr loss:{avg_loss:.4f}, cme_loss={cme_avg:.4f}")
                loss.backward()

                self.csr_optim.step()

            if val_loader is not None:
                with torch.no_grad():
                    self.tscn.eval()

                    iterator = tqdm.tqdm(val_loader)
                    val_loss = 0
                    for data, label in iterator:
                        label_m, label_p = label.abs().type(torch.FloatTensor).to(self.device), label.angle().type(
                            torch.FloatTensor).to(
                            self.device)

                        label_r = label.real.type(torch.FloatTensor).to(self.device)
                        label_i = label.imag.type(torch.FloatTensor).to(self.device)

                        cme_mag, output = self.tscn(data)
                        real, imag = output.real, output.imag
                        mag, phase = output.abs(), output.angle()

                        cme_loss = self.loss_weight_coefficient * nn.MSELoss()(cme_mag, label_m)
                        loss = nn.MSELoss()(mag, label_m) + nn.MSELoss()(real, label_r) + nn.MSELoss()(imag,
                                                                                                       label_i) + cme_loss
                        val_loss += loss.item() / len(val_loader)
                        iterator.set_description(
                            f"epoch:{epoch} \t validation loss:{val_loss:.4f}, batch_loss={loss.item():.4f}")

            if avg_loss <= best_loss:
                if self.multi:
                    best_csr = deepcopy(self.tscn.csr.module.state_dict())
                    best_cme = deepcopy(self.tscn.cme.module.state_dict())
                else:
                    best_csr = deepcopy(self.tscn.csr.state_dict())
                    best_cme = deepcopy(self.tscn.cme.state_dict())
                best_loss = avg_loss

        torch.save(best_cme, os.path.join(self.weight_pth, self.cme_filename))
        torch.save(best_csr, os.path.join(self.weight_pth, self.csr_filename))

        return best_cme, best_csr

    def inference(self, src_path, dst_path):
        def split_infer(filename, sec, sr=16000, cutoff=True):
            data, sr = librosa.load(filename, sr=sr)

            batches = []

            i = 0
            padded = 0
            batch_size = sr * sec
            while i < len(data):
                if cutoff:
                    if i + batch_size <= len(data):
                        batches.append(data[i:i + batch_size])
                    else:
                        pass
                else:
                    if i + batch_size <= len(data):
                        batches.append(data[i:i + batch_size])
                    else:
                        batch = data[i:len(data)]
                        batches.append(batch)
                i = i + batch_size

            return batches

        model = TSCN_Module(device=self.device).to(self.device)
        model.cme.load_state_dict(torch.load(os.path.join(self.weight_pth, self.cme_filename), map_location=self.device))
        model.csr.load_state_dict(torch.load(os.path.join(self.weight_pth, self.csr_filename), map_location=self.device))

        infer_signal = []
        noisy_batches = split_infer(src_path.strip(), sr=self.sr, sec=self.sec, cutoff=self.cutoff)

        for batch in noisy_batches:
            if len(batch) == 0:
                continue
            signal = torch.tensor(batch)
            signal = Spectrogram(n_fft=self.n_fft, win_length=self.win_len, power=None, return_complex=True)(signal)
            signal = torch.unsqueeze(signal, dim=0).to(self.device)

            cme_mag, output = model(signal)

            wave = torch.istft(output, n_fft=self.n_fft, hop_length=int(self.win_len/2), win_length=self.win_len, length=len(batch))
            wave = torch.flatten(wave).cpu()
            wave = wave.detach().numpy()
            wave = wave / np.max(wave)
            infer = wave.tolist()
            infer_signal.extend(infer)

        write(dst_path, self.sr, np.array(infer_signal).astype(np.float32))
