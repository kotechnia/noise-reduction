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
        sd_checker = re.sub('[^VN]', '', sd_file_name)

        if sd_checker == 'VN':
            sn_file_name = re.sub('VN', 'NV', sd_file_name)
            sn_file_path = os.path.join(dir_path, sn_file_name)

            if os.path.isfile(sd_file_path) == False:
                continue
            if os.path.isfile(sn_file_path) == False:
                continue

            sd_files.append(sd_file_path)
            sn_files.append(sn_file_path)
            
    df = pd.DataFrame(zip(sd_files, sn_files), columns = ['sd_file_path', 'sn_file_path'])
    return df

def filename_to_key(path, sep='_', start=0, end=-1):
    filename = os.path.basename(path)
    filename, ext = os.path.splitext(filename)
    key = filename.split(sep)[start:end]
    key = sep.join(key)
    return key

def file_match(directory_path, clean_name = '**/**_VN.wav', noisy_name='**/**_NV.wav', script_name='**/**_VN.json'):
    
    clean_list = glob(os.path.join(directory_path, clean_name), recursive=True)
    noisy_list = glob(os.path.join(directory_path, noisy_name), recursive=True)
    script_list = glob(os.path.join(directory_path, script_name), recursive=True)

    df_clean_list = pd.DataFrame({'key':map(filename_to_key, clean_list), 'clean_path':clean_list})
    df_noisy_list = pd.DataFrame({'key':map(filename_to_key, noisy_list), 'noisy_path':noisy_list})
    df_script_list = pd.DataFrame({'key':map(filename_to_key, script_list), 'script_path':script_list})

    df = pd.merge(df_clean_list, df_noisy_list, on=['key'])
    df = pd.merge(df, df_script_list, on=['key'])
    
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
        "--dataset_root",
        default='share',
        help=""" default : 'share' """
    )

    parser.add_argument(
        "--csv_save_path",
        default='share/dataset.csv',
        help=""" default : 'share/dataset.csv' """
    )
    
    args = parser.parse_args()
    print('setting parameters')
    if args.dataset_root:print('dataset_root : {}'.format(args.dataset_root))
    if args.csv_save_path:print('csv_save_path : {}'.format(args.csv_save_path))
    
    df = file_match(args.dataset_root)
    df = train_val_test_shuffle(df)
    df.to_csv(args.csv_save_path, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    tvt, counts = np.unique(df['train_val_test'], return_counts=True)
    print(pd.DataFrame({"train_val_test":tvt, "count":counts}))
    
if __name__ == '__main__':
    main()
