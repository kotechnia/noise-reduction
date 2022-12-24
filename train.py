def main():
    from tscn.dataset import TscnLoader
    from tscn.TSCN import TSCN
    from utils.utils import DivideWAVFile
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import argparse
    import shutil
    import os
    
    parser = argparse.ArgumentParser(description='tscn helper.')

    parser.add_argument(
        "--model",
        default='./models/tscn',
        help="""'path of the model' default : --model='./models/rnnoise'"""
    )

    parser.add_argument(
        "--csv_file",
        default='./share/dataset.csv',
        help="""default='./share/dataset.csv'"""
    )

    parser.add_argument(
        "--cme_epochs", 
        default=None,
        help="""'model cme_epochs' default : --cme_epochs=None"""
    )
    
    parser.add_argument(
        "--finetune_epochs", 
        default=None,
        help="""'model finetune_epochs' default : --finetune_epochs=None"""
    )
    
    parser.add_argument(
        "--csr_epochs", 
        default=None,
        help="""'model csr_epochs' default : --csr_epochs=None"""
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        help="""'model batch_size' default : --batch_size=8"""
    )
    
    parser.add_argument(
        "--multi_gpu",
        default=True,
        help="""'model multi_gpu' default : --multi_gpu=True"""
    )
    
    parser.add_argument(
        "--preproc_path",
        default=None,
        help="""'preproc_path' default : None"""
    )
    
    args = parser.parse_args()
    print('tscn setting parameters')
    if args.model:print('model : {}'.format(args.model))
    if args.csv_file:print('csv_file : {}'.format(args.csv_file))     
    if args.cme_epochs:print('cme_epochs : {}'.format(args.cme_epochs))
    if args.finetune_epochs:print('finetune_epochs : {}'.format(args.finetune_epochs))
    if args.csr_epochs:print('csr_epochs : {}'.format(args.csr_epochs))
    if args.batch_size:print('batch_size : {}'.format(args.batch_size))
    if args.multi_gpu:print('multi_gpu : {}'.format(args.multi_gpu))
    if args.preproc_path:print('preproc_path : {}'.format(args.preproc_path))
    
    
    #df_global = pd.read_csv(args.csv_file, encoding='euc-kr')    
    df_global = pd.read_csv(args.csv_file, encoding='utf-8-sig')    
    df_train = df_global.loc[np.where(df_global['train_val_test'] == 'TR')].reset_index(drop=True)
    df_valid = df_global.loc[np.where(df_global['train_val_test'] == 'VA')].reset_index(drop=True)

    traincsv = 'share/temp/trainset.csv'
    validcsv = 'share/temp/validset.csv'
    
    print('make train set')
    tt = DivideWAVFile(df=df_train,
                       output_dir='share/temp/',
                       csv_file_path=traincsv,
                       abspath=True, ignore_remainder=0.0,tqdm=tqdm)
    [_ for _, _ in tt]
    
    print('make valid set')
    tt = DivideWAVFile(df=df_valid,
                       output_dir='share/temp/',
                       csv_file_path=validcsv,
                       abspath=True, ignore_remainder=0.0,tqdm=tqdm)
    [_ for _, _ in tt]
    
    continue_flag = False
    
    train_tscnloader = TscnLoader(
        path=traincsv,
        batch_size=int(args.batch_size),
        save_preprocess_path=args.preproc_path
    )
    
    valid_tscnloader = TscnLoader(
        path=validcsv,
        batch_size=int(args.batch_size),
        save_preprocess_path=args.preproc_path
    )
    
    tscn_model = TSCN(
        weight_pth=args.model,
        transfer=continue_flag,
        multi=args.multi_gpu,
        device="cuda",
    )
    
    tscn_model.fit(
        train_loader=train_tscnloader,
        val_loader=valid_tscnloader,
        cme_epochs=(args.cme_epochs),
        finetuning_epochs=(args.finetune_epochs),
        csr_epochs=(args.csr_epochs)
    )
    
    continue_flag=True
    
    shutil.rmtree('share/temp')

if __name__ == '__main__':
    main()    
