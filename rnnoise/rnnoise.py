#!/usr/bin/python

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
import numpy as np
import pandas as pd
import os
import subprocess
import shutil
import stat


#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}
    
def model_build_and_compile(
    reg = 0.000001,
    constraint=WeightClip(0.499),
    model_path = "./rnnoise",
    continue_flag = False
):
    
    model_save_path = os.path.join(model_path, 'training/weights.hdf5')
    
    print('Build model...')
    main_input = Input(shape=(None, 42), name='main_input')
    tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
    vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
    vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
    noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
    noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
    denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

    denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

    denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

    model.compile(loss=[mycost, my_crossentropy],
                  metrics=[msse],                  
                  optimizer='adam', loss_weights=[10, 0.5])
    
    if continue_flag:
        try:
            print(model_save_path)
            print('기존 모델 정보 획득')
            model.load_weights(model_save_path)
        except:
            print("Failed to load model")
    
    return model

def load_train_feature_data(path):
    print('Loading data...')
    with h5py.File(path, 'r') as hf:
        all_data = hf['data'][:]
    print('done.')
    return all_data

def dataset_split(all_data, window_size):
    
    nb_sequences = len(all_data)//window_size
    print(nb_sequences, ' sequences')
    
    x_train = all_data[:nb_sequences*window_size, :42]
    x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

    y_train = np.copy(all_data[:nb_sequences*window_size, 42:64])
    y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

    noise_train = np.copy(all_data[:nb_sequences*window_size, 64:86])
    noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

    vad_train = np.copy(all_data[:nb_sequences*window_size, 86:87])
    vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

    all_data = 0;
    print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)
    
    return x_train, y_train, noise_train, vad_train

def get_sd_sn(dataloader):
    sds = []
    sns = []
    #for sd, sn in dataloader:
    for idx, sd, sn in dataloader:
        sds.append(sd)
        sns.append(sn)
    return sds, sns

def ffmpeg_format_text(path, file_list):
    with open(path, 'w') as f:
        for name in file_list:
            f.write("file '"+name+"'\n")
            
def create_data(dataloader, proc_type, model_path = './rnnoise'):
    
    current_path = os.getcwd()
    os.chdir( model_path )
    
    sds, sns = get_sd_sn(dataloader)
    
    if proc_type == 'TE':
        ffmpeg_format_text('./src/train_clean.txt', sds)
        ffmpeg_format_text('./src/train_noisy.txt', sns)
        subprocess.call(['./dataset_train.sh'])
        feature_data = load_train_feature_data('./training/train_training.h5')
        
    elif proc_type == 'VA':
        ffmpeg_format_text('./src/valid_clean.txt', sds)
        ffmpeg_format_text('./src/valid_noisy.txt', sns)
        subprocess.call(['./dataset_valid.sh'])
        feature_data = load_train_feature_data('./training/valid_training.h5')
        
    os.chdir(current_path)
        
    window_size = 2000
        
    return dataset_split(feature_data, window_size)


def demo(src_path, dst_path, model_path = './rnnoise'):
    
    model_call_path = os.path.join(model_path, 'examples/rnnoise_demo')
    temp_n_path = os.path.join(model_path, 'temp_n.raw')
    temp_d_path = os.path.join(model_path, 'temp_d.raw')
    
    #'-loglevel', 'quiet' 출력문 제거
    subprocess.call(['ffmpeg', '-i', src_path, '-ac', '1', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '48000', '-y', temp_n_path, '-loglevel', 'quiet']) #2.88mb
    #subprocess.call(['./examples/rnnoise_demo', 'temp_n.raw', 'temp_d.raw'])
    subprocess.call([model_call_path, temp_n_path, temp_d_path])
    subprocess.call(['ffmpeg', '-f', 's16le', '-ar', '48000', '-ac', '1', '-i', temp_d_path, '-y', dst_path, '-loglevel', 'quiet']) #2.88mb
    
    
def make_demo(model_path = './rnnoise'):
    current_path = os.getcwd()
    os.chdir(model_path)
    print('build demo path : {}'.format(model_path))
    subprocess.call(['./build_demo.sh'])
    os.chdir(current_path)
    
def file_remove(path):
    if os.path.isfile(path):
        os.remove(path)
    
def copytree(src, dst, symlinks = False, ignore = None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)    
    
class RNNoise():
    def __init__(self,
                 model_path = None,
                 train_dataloader = None,
                 valid_dataloader = None,
                 batch_size=64,
                 epochs=10,
                 continue_flag = True,
                 save_flag = True,
                ):
        
        self.iter = 0
        
        self.train_flag = False
        self.continue_flag = continue_flag
        
        if model_path is None:
            self.model_path = model_path
            self.model = model_build_and_compile(continue_flag = False)
            
        else:
            self.model_path = model_path
            
            if os.path.isdir(self.model_path) == False:
                print('create model')
                
                copy_path = './rnnoise/rnnoise_copy'
                
                if os.path.isdir(copy_path) == False:
                    raise Exception('copy_path error')
                    
                copytree(copy_path, self.model_path, symlinks=True)
                #copytree('/home/jaewon/work/jupyter9102-test/dns/rnnoise/rnnoise_copy', self.model_path, symlinks=True)
            
            self.model = model_build_and_compile(continue_flag = continue_flag, model_path = model_path)
            
            if train_dataloader is not None and valid_dataloader is not None:
                self.train(train_dataloader, valid_dataloader, batch_size, epochs)
                
                if save_flag:
                    self.save(self.model_path)
                
            
    def train(self,
              train_dataloader,
              valid_dataloader,
              batch_size,
              epochs
              ):
        
        train_x, train_y, train_noise, train_vad = create_data(train_dataloader, 'TE', model_path = self.model_path)
        valid_x, valid_y, valid_noise, valid_vad = create_data(valid_dataloader, 'VA', model_path = self.model_path)
        
        self.history = self.model.fit(
            train_x, [train_y, train_vad],
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (valid_x, [valid_y, valid_vad])
        )
        
        self.iter += 1
        
        self.train_flag = True
        
    def save(self,
             model_path=None
            ):
        
        if model_path is not None:
            if self.model_path != model_path:
                copytree(self.model_path, model_path, symlinks=True)
            
            self.model_path = model_path
        else:
            pass
        
        self._clean_memory()
        
        if self.train_flag:
            
            self.loss_path = os.path.join(self.model_path, 'training/loss/loss_history.csv')
            print(self.loss_path)
            #try:
            #    with open(self.loss_path, mode='w') as f:
            #        pd.DataFrame(self.history.history).to_csv(f, index=False)
            #except:
            #    pass
            
            # 2021-11-01 loss - history 추가형태로기록
            if self.continue_flag:
                if os.path.isfile(self.loss_path):
                    pd.DataFrame(self.history.history).to_csv(self.loss_path, mode='a', header=False, index=False)
                else:
                    pd.DataFrame(self.history.history).to_csv(self.loss_path, mode='w', header=True, index=False)
            else:
                pd.DataFrame(self.history.history).to_csv(self.loss_path, mode='w', header=True, index=False)
            

            self.model_save_path = os.path.join(self.model_path, 'training/weights.hdf5')  
            self.model.save(self.model_save_path)
            
            make_demo(model_path = self.model_path)
            
            #self._clean_memory()
            
        
    def denoise(self,
                src_path, dst_path
               ):
    
        demo(src_path, dst_path, model_path = self.model_path)
        
        
    def denoise_dataloader(
        self,
        output_dir = './temp_test',
        dataloader = None,
    ):
        
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame([], columns=['clean', 'noisy', 'denoise'])

        for index, sd_file_path, sn_file_path in dataloader:

            sn_file_name = sn_file_path.split('/')[-1]
            denoise_path = os.path.join(output_dir, sn_file_name)

            self.denoise(
                src_path = sn_file_path,
                dst_path = denoise_path,
            )

            data = {'clean':sd_file_path, 'noisy':sn_file_path, 'denoise':denoise_path}
            df = df.append(pd.DataFrame([data]), ignore_index=True)

        df.to_excel(os.path.join(output_dir, 'info.xlsx'), index=False)
        
        return df

    
    def _clean_memory(self):
        file_remove(os.path.join(self.model_path, 'src/train_clean.txt'))
        file_remove(os.path.join(self.model_path, 'src/train_clean.raw'))
        file_remove(os.path.join(self.model_path, 'src/train_noisy.txt'))
        file_remove(os.path.join(self.model_path, 'src/train_noisy.raw'))
        file_remove(os.path.join(self.model_path, 'src/train_training.f32'))

        file_remove(os.path.join(self.model_path, 'src/valid_clean.txt'))
        file_remove(os.path.join(self.model_path, 'src/valid_clean.raw'))
        file_remove(os.path.join(self.model_path, 'src/valid_noisy.txt'))
        file_remove(os.path.join(self.model_path, 'src/valid_noisy.raw'))
        file_remove(os.path.join(self.model_path, 'src/valid_training.f32'))

        file_remove(os.path.join(self.model_path, 'training/train_training.h5'))
        file_remove(os.path.join(self.model_path, 'training/valid_training.h5'))