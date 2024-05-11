import pandas as pd
import numpy as np


class Transforms:

    def sma(
        self, X: pd.DataFrame, N: int, variable: str, period: str = "D"
    ) -> pd.DataFrame:
        X[f"{variable}_SMA_{N}{period}"] = X.groupby("ID")[f"{variable}"].transform(
            lambda x: x.rolling(window=N).mean().shift()
        )
        return X

    def ewma(
        self, X: pd.DataFrame, N: int, variable: str, period: str = "D"
    ) -> pd.DataFrame:
        X[f"{variable}_EWMA_{N}{period}"] = X.groupby("ID")[f"{variable}"].transform(
            lambda x: x.ewm(span=N).mean().shift()
        )
        return X

    def roc(self, X: pd.DataFrame, variable: str, N: int = 14) -> pd.DataFrame:
        X[f"{variable}_ROC_{N}D"] = X.groupby("ID")[variable].transform(
            lambda x: x.diff(N) / x.shift(N) * 100
        )
        return X

    def log_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        X["RETURNS"] = X.groupby("ID")["CLOSE"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        return X
