import os
import pandas as pd

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
        
        df.rename(columns={'clean':'sd_file_path', 'noisy':'sn_file_path'}, inplace=True)
        
        if 'sd_file_path' not in df.columns:
            raise Exception('DataLoader Column Error')
            
        if 'sn_file_path' not in df.columns:
            raise Exception('DataLoader Column Error')
        
        self.df = df
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return [self.df.loc[i, 'sd_file_path'], self.df.loc[i, 'sn_file_path']]
    
    def __iter__(self):
        for i in range(len(self.df)):
            yield [self.df.loc[i, 'sd_file_path'],
                   self.df.loc[i, 'sn_file_path']]
