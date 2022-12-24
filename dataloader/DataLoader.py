import os
import pandas as pd


__CLEAN_COLUMN__='clean_path'
__NOISY_COLUMN__='noisy_path'

class DataLoader():
    
    def __init__(self, 
                 df=None,
                 path=None
                ):
        
        if df is None:
            if path is None:
                raise Exception('DataLoader Parameter Error')
            else:
                _, ext = os.path.splitext(path)
                
                if ext == '.csv':
                    df = pd.read_csv(path)
                elif ext == '.xlsx':
                    df = pd.read_excel(path)
        
        
        df.rename(columns={'clean':__CLEAN_COLUMN__, 'noisy':__NOISY_COLUMN__}, inplace=True)
        df.rename(columns={'sd_file_path':__CLEAN_COLUMN__, 'sn_file_path':__NOISY_COLUMN__}, inplace=True)
        
        
        if __CLEAN_COLUMN__ not in df.columns:
            raise Exception('DataLoader Column Error')
            
        if __NOISY_COLUMN__ not in df.columns:
            raise Exception('DataLoader Column Error')
        
        self.df = df
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return [self.df.loc[i, __CLEAN_COLUMN__], self.df.loc[i, __NOISY_COLUMN__]]
    
    def __iter__(self):
        for i in range(len(self.df)):
            yield [self.df.loc[i, __CLEAN_COLUMN__],
                   self.df.loc[i, __NOISY_COLUMN__]]
