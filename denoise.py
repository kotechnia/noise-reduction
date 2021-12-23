def main():
    from tscn.TSCN import TSCN
    from utils.utils import calc_estoi
    from dataloader.DataLoader import DataLoader
    from utils.utils import denoise_estoi
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import argparse
    
    parser = argparse.ArgumentParser(description='tscn helper.')

    parser.add_argument(
        "--model",
        default='./models/tscn',
        help="""'path of the model' default : --model='./models/tscn'"""
    )

    parser.add_argument(
        "--noisy",
        default=None,
        help="""'path of sn file' example : --noisy='./share/sn_file.wav'"""
    )

    parser.add_argument(
        "--denoise",
        default=None,
        help="""'path of sn file' example : --denoise='./share/de_file.wav'"""
    )
    
    parser.add_argument(
        "--clean",
        default=None,
        help="""'path of sn file' example : --clean='./share/sd_file.wav'"""
    )
    
    parser.add_argument(
        "--csv_file",
        default='./share/dataset.csv',
        help="""default : --csv_file='./share/dataset.csv'"""
    )

    parser.add_argument(
        "--output_dir",
        default='./share/denoise/',
        help="""default : --output_dir='./share/denoise/'"""
    )
    
    
    args = parser.parse_args()
    print('tscn setting parameters')
    if args.model:print('model : {}'.format(args.model))
    if args.noisy:print('noisy : {}'.format(args.noisy))    
    if args.denoise:print('denoise : {}'.format(args.denoise)) 
    if args.clean:print('clean : {}'.format(args.clean)) 
    
    if args.noisy is None:
        if args.csv_file:print('dataset : {}'.format(args.csv_file))
        if args.output_dir:print('output_dir : {}'.format(args.output_dir))
    
    tscn_model = TSCN(
        weight_pth = args.model,
        transfer=True,
        device="cuda",
    )
    
    if args.noisy:        
        
        tscn_model.inference(args.noisy, args.denoise)

        if args.clean:
            estoi = calc_estoi(args.denoise, args.clean)
            print(f'estoi : {estoi}')
        else:
            print(f'')
            
    else:
        
        df_global = pd.read_csv(args.csv_file, encoding='euc-kr')    
        df_test = df_global.loc[np.where(df_global['train_val_test'] == 'TE')].reset_index(drop=True)
        
        test_dataloader = DataLoader(df=df_test)

        df_estoi_data = denoise_estoi(
            tscn_model.inference,
            test_dataloader,
            output_dir = args.output_dir,
            tqdm=tqdm
        )

        print('estoi mean = {}'.format(df_estoi_data['estoi'].mean()))
        
    
        

if __name__ == '__main__':
    main()    