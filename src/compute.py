import pandas as pd
import numpy as np

def compute_signal(M:pd.DataFrame, strategy:str = 'momentum') -> pd.DataFrame:
    if strategy == 'momentum':
        M['signal'] = np.where(M['CLOSE_SMA_9D'] > M['CLOSE_SMA_21D'], 1, 0)    # long signal
        M['signal'] = np.where(M['CLOSE_SMA_9D'] < M['CLOSE_SMA_21D'], -1, M['signal'])     # short signal
    
    return M

def pivotToClose(df:pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
        df['DATE'] = pd.to_datetime(df['DATE'])

    # Pivot the DataFrame to wide format
    transformed_df = df.pivot(index='DATE', columns='ID', values='CLOSE')

    return transformed_df

def getData(df:pd.DataFrame, stocks:list[str], start:pd.DatetimeIndex, end:pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.loc[df.ID.isin(stocks)].loc[df.DATE > start].loc[df.DATE < end]
    stockData = X[['ID','DATE','CLOSE']]
    stockData = pivotToClose(stockData)
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

class Wrangle():
    def __init__(self) -> None:
        pass

    def get_stock(self):
        pass

class Portfolio():
    def __init__(self) -> None:
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
