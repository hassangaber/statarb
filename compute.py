import pandas as pd
import numpy as np

def compute_signal(M:pd.DataFrame, strategy:str = 'momentum') -> pd.DataFrame:
    if strategy == 'momentum':
        M['signal'] = np.where(M['CLOSE_SMA_9D'] > M['CLOSE_SMA_21D'], 1, 0)    # long signal
        M['signal'] = np.where(M['CLOSE_SMA_9D'] < M['CLOSE_SMA_21D'], -1, M['signal'])     # short signal
    
    return M

class Kernel():
    def __init__(self) -> None:
        pass

    def get_stock(self):
        pass

class Transforms():
    def __init__(self):
        pass

    def sma(self, X:pd.DataFrame, N:int, variable:str, period:str='D') -> pd.DataFrame:
        X[f'{variable}_SMA_{N}{period}'] = X.groupby('ID')[f'{variable}'].transform(lambda x: x.rolling(window=N).mean().shift())
        return X

    def ewma(self, X:pd.DataFrame, N:int, variable:str, period:str='D') -> pd.DataFrame:
        X[f'{variable}_EWMA_{N}{period}'] = X.groupby('ID')[f'{variable}'].transform(lambda x: x.ewm(span=N).mean().shift())
        return X
    
    def roc(self, X:pd.DataFrame, variable:str, N:int=14) -> pd.DataFrame:
        X[f'{variable}_ROC_{N}D'] = X.groupby('ID')[variable].transform(lambda x: x.diff(N) / x.shift(N) * 100)
        return X
