import pandas as pd
import numpy as np


class Extract:

    @staticmethod
    def pivotToClose(df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"])
        transformed_df = df.pivot(index="DATE", columns="ID", values="CLOSE")
        return transformed_df

    @staticmethod
    def stockSpread(
        df: pd.DataFrame,
        stocks: list[str],
        start: pd.DatetimeIndex,
        end: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        stockData = df[
            df["ID"].isin(stocks) & (df["DATE"] > start) & (df["DATE"] < end)
        ]
        stockDataPivoted = stockData.pivot(index="DATE", columns="ID", values="CLOSE")
        log_returns = np.log(stockDataPivoted / stockDataPivoted.shift(1))
        log_returns = log_returns.dropna()
        meanReturns = log_returns.mean()
        covMatrix = log_returns.cov()
        return (log_returns, meanReturns, covMatrix)
