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
    model_path = "./training/weights.hdf5",
    continue_flag = False
):
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
        model.load_weights(model_path)
    
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
    for sd, sn in dataloader:
        sds.append(sd)
        sns.append(sn)
    return sds, sns

def ffmpeg_format_text(path, file_list):
    with open(path, 'w') as f:
        for name in file_list:
            f.write("file '"+name+"'\n")
            
def create_data(dataloader, proc_type):
    if proc_type == 'TE':
        sds, sns = get_sd_sn(dataloader)
        ffmpeg_format_text('rnnoise/src/train_clean.txt', sds)
        ffmpeg_format_text('rnnoise/src/train_noisy.txt', sns)
        
        os.chdir( './rnnoise' )
        subprocess.call(['./dataset_train.sh'])
        
        feature_data = load_train_feature_data('./training/train_training.h5')
        os.chdir( './../' )
        
    elif proc_type == 'VA':
        sds, sns = get_sd_sn(dataloader)
        ffmpeg_format_text('rnnoise/src/valid_clean.txt', sds)
        ffmpeg_format_text('rnnoise/src/valid_noisy.txt', sns)
    
        os.chdir( './rnnoise' )
        subprocess.call(['./dataset_valid.sh'])
        
        feature_data = load_train_feature_data('./training/valid_training.h5')
        os.chdir( './../' )
        
    window_size = 2000
        
    return dataset_split(feature_data, window_size)


def demo(src_path, dst_path):
    #'-loglevel', 'quiet' 출력문 제거
    subprocess.call(['ffmpeg', '-i', src_path, '-ac', '1', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '48000', '-y', './rnnoise/temp_n.raw', '-loglevel', 'quiet']) #2.88mb
    #subprocess.call(['./examples/rnnoise_demo', 'temp_n.raw', 'temp_d.raw'])
    subprocess.call(['./rnnoise/examples/rnnoise_demo', './rnnoise/temp_n.raw', './rnnoise/temp_d.raw'])
    subprocess.call(['ffmpeg', '-f', 's16le', '-ar', '48000', '-ac', '1', '-i', './rnnoise/temp_d.raw', '-y', dst_path, '-loglevel', 'quiet']) #2.88mb
    
    
def make_demo():
    os.chdir('./rnnoise')
    subprocess.call(['./build_demo.sh'])
    os.chdir('./..')