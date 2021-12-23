import os
import pandas as pd
import numpy as np
import re
from glob import glob

def sd_sn_file_match(directory_path, wav_file = '**/**.wav'):
    file_list = glob(os.path.join(directory_path, wav_file), recursive=True)
    
    sd_files = []
    sn_files = []

    for i in range(len(file_list)):
        sd_file_path = file_list[i]
        dir_path, sd_file_name = os.path.split(sd_file_path)
        sd_checker = re.sub('[^SD]', '', sd_file_name)

        if sd_checker == 'SD':
            sn_file_name = re.sub('SD', 'SN', sd_file_name)
            sn_file_path = os.path.join(dir_path, sn_file_name)

            if os.path.isfile(sd_file_path) == False:
                continue
            if os.path.isfile(sn_file_path) == False:
                continue

            sd_files.append(sd_file_path)
            sn_files.append(sn_file_path)
            
    df = pd.DataFrame(zip(sd_files, sn_files), columns = ['sd_file_path', 'sn_file_path'])
    return df

def train_val_test_shuffle(df, train_ratio = 0.8, test_ratio = 0.1):
    data_length = len(df)
    indices = np.array(list(range(data_length)))

    np.random.shuffle(indices)

    train_indices = indices[:int(data_length*train_ratio)]
    valid_indices = indices[int(data_length*train_ratio):int(data_length*(1-test_ratio))]
    test_indices = indices[int(data_length*(1-test_ratio)):]

    df.loc[train_indices, 'train_val_test']='TR'
    df.loc[valid_indices, 'train_val_test']='VA'
    df.loc[test_indices, 'train_val_test']='TE'

    #df = df.loc[indices].reset_index(drop=True)
    
    return df

def main():
    from dataloader.DataLoader import DataLoader
    import argparse
    
    parser = argparse.ArgumentParser(description='make csv')

    parser.add_argument(
        "--wav_files_dir",
        default='share/data/',
        help=""" default : 'share/data/' """
    )

    parser.add_argument(
        "--csv_save_path",
        default='share/dataset.csv',
        help=""" default : 'share/dataset.csv' """
    )
    
    args = parser.parse_args()
    print('setting parameters')
    if args.wav_files_dir:print('wav_files_dir : {}'.format(args.wav_files_dir))
    if args.csv_save_path:print('csv_save_path : {}'.format(args.csv_save_path))
    
    df = sd_sn_file_match(args.wav_files_dir)
    df = train_val_test_shuffle(df)
    
    df.to_csv(args.csv_save_path, mode='w', header=True, index=False, encoding='euc-kr')
    
    
if __name__ == '__main__':
    main()