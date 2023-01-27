from espnet_asr.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
from datetime import datetime
from dataloader.DataLoader import __CLEAN_COLUMN__, __NOISY_COLUMN__
import json
import os
import re
from difflib import SequenceMatcher
import argparse
import torch

def json_open(path):
    with open(path) as f:
        return json.load(f)

def current_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def text_clean(text):
    text = re.sub('\[unk\]', '', text) 
    text = re.sub('\[un', '', text) 
    text = re.sub('\[u', '', text) 
    text = re.sub('\[', '', text) 
    text = re.sub("[ ]+", " ", text)
    return text

def transcription_selection(text, select='left'):
    if select == 'left':
        pattern = '/.+'
    elif select ==  'right':
        pattern = '.+/'
        
    find_pattern = re.findall('\([^\(\)/]+\)?/\(?[^\(\)/]+\)', text)
    
    for p in find_pattern:
        refine = re.sub(pattern, '', p)
        refine = re.sub('[\(\)]', '', refine)
        
        text = text.replace(p, refine)
        
    return text
    

def main():

    parser = argparse.ArgumentParser(description='tscn helper.')

    parser.add_argument(
        "--dataset_path",
        default='./share/dataset.csv',
    )

    parser.add_argument(
        "--denoise_dir",
        default='./share/denoise',
    )

    parser.add_argument(
        "--results_path",
        default=None,
    )
    
    parser.add_argument(
        "--device",
        default='cpu',
    )

    args = parser.parse_args()


    dataset_path = args.dataset_path
    denoise_path = os.path.join(args.denoise_dir, 'info.csv')

    print(f"{current_time()}  load model")
    d = ModelDownloader("espnet_asr/.cache/espnet")
    o = d.download_and_unpack("espnet_asr/mdl/ksponspeech.zip")

    speech2text = Speech2Text(
        o['asr_train_config'],
        o['asr_model_file'],
        device=args.device
    )
    print(f"{current_time()} ~load model")

    df_dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    df_dataset = df_dataset.loc[np.where(df_dataset['train_val_test']=='TE')[0]].reset_index(drop=True)
    df_denoise = pd.read_csv(denoise_path, encoding='utf-8-sig')

    df_dataset = pd.merge(df_dataset, df_denoise, on=[__CLEAN_COLUMN__,__NOISY_COLUMN__])
    
    try:
        df_dataset.insert(df_dataset.shape[1], 'f1_noisy', 0)
    except:
        pass
    try:
        df_dataset.insert(df_dataset.shape[1], 'f1_denoise', 0)
    except:
        pass
    try:
        df_dataset.insert(df_dataset.shape[1], 'f1_error_rate', 0)
    except:
        pass

    for i in tqdm(range(len(df_dataset)), ascii=True):
        texts = []

        script_path = df_dataset.loc[i, 'script_path']
        script_data = json_open(script_path)
        dialogs = script_data.pop('dialogs')

        for column in [ __NOISY_COLUMN__ ]:

            file_path = df_dataset.loc[i, column]
            tqdm.write(file_path)

            text_path = os.path.splitext(file_path)[0]+'.txt'

            if False:
                all_text = ""
                with open(text_path, 'r') as f:
                    all_text = f.read()

            else:
                audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
                all_text = [] 
                for dialog in dialogs:
                    start = int(dialog['startTime'] * sr)
                    end = int(dialog['endTime'] * sr)
                    segment_audio = audio[start:end]
                    if segment_audio.shape[0] == 0 : continue
                    if segment_audio.shape[0] > 180 * sr : continue
                    outputs = speech2text(segment_audio)
                    for text, token, token_int, hypotesis in outputs:
                        all_text.append(text)
                    torch.cuda.empty_cache()

                all_text = text_clean(' '.join(all_text)).strip()

                with open(text_path, 'w') as f:
                    f.write(all_text)

            tqdm.write(f"{current_time()} {all_text}")
            texts.append(all_text)

        for column in [ 'denoise_path' ]:

            file_path = df_dataset.loc[i, column]
            tqdm.write(file_path)

            text_path = os.path.splitext(file_path)[0]+'.txt'

            if False:
                all_text = ""
                with open(text_path, 'r') as f:
                    all_text = f.read()

            else:
                audio, sr = librosa.load(file_path, sr=16000, mono=True)

                all_text = []
                for dialog in dialogs:
                    start = int(dialog['startTime'] * sr)
                    end = int(dialog['endTime'] * sr)
                    segment_audio = audio[start:end]
                    if segment_audio.shape[0] == 0 : continue
                    if segment_audio.shape[0] > 180 * sr : continue
                    outputs = speech2text(segment_audio)
                    for text, token, token_int, hypotesis in outputs:
                        all_text.append(text)
                    torch.cuda.empty_cache()

                all_text = text_clean(' '.join(all_text)).strip()

                with open(text_path, 'w') as f:
                    f.write(all_text)

            tqdm.write(f"{current_time()} {all_text}")
            texts.append(all_text)


        for column in [ __CLEAN_COLUMN__ ]:

            file_path = df_dataset.loc[i, column]
            tqdm.write(file_path)

            text_path = os.path.splitext(file_path)[0]+'.txt'

            if False:
                all_text = ""
                with open(text_path, 'r') as f:
                    all_text = f.read()

            else:
                audio, sr = librosa.load(file_path, sr=16000, mono=True)

                all_text = []
                for dialog in dialogs:
                    start = int(dialog['startTime'] * sr)
                    end = int(dialog['endTime'] * sr)
                    if segment_audio.shape[0] == 0 : continue
                    if segment_audio.shape[0] > 180 * sr : continue
                    text = dialog['speakerText']
                    all_text.append(text)
                    torch.cuda.empty_cache()

                all_text = transcription_selection(' '.join(all_text)).strip()

                with open(text_path, 'w') as f:
                    f.write(all_text)

            tqdm.write(f"{current_time()} {all_text}")
            texts.append(all_text)


        f1_noisy = SequenceMatcher(isjunk=None, a=texts[0], b=texts[2]).ratio()
        f1_denoise = SequenceMatcher(isjunk=None, a=texts[1], b=texts[2]).ratio()

        df_dataset.loc[i, 'f1_noisy'] = f1_noisy
        df_dataset.loc[i, 'f1_denoise'] = f1_denoise
        df_dataset.loc[i, 'f1_error_rate'] = (f1_denoise - f1_noisy) / (1 - f1_noisy)

    df_dataset.to_csv(args.results_path, index=False, encoding='utf-8-sig')

    f1_noisy = df_dataset['f1_noisy'].mean()
    f1_denoise = df_dataset['f1_denoise'].mean()
    f1_score_error_rate = (f1_denoise - f1_noisy) / (1 - f1_noisy)

    print(f"{current_time()} f1-score-error-rate : {f1_score_error_rate}")


if __name__ == '__main__':
    main()
