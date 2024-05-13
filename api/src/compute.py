import pandas as pd
import numpy as np


def breakout_probabilities(
    portfolio_sims: np.array, upper_limit: float, lower_limit: float
) -> tuple[float, float]:
    """Calculate the probabilities of breaking out above an upper limit or below a lower limit."""
    total_sims = portfolio_sims.shape[0]
    upper_breakouts = np.sum(portfolio_sims.max(axis=0) > upper_limit)
    lower_breakouts = np.sum(portfolio_sims.min(axis=0) < lower_limit)
    return (upper_breakouts / total_sims, lower_breakouts / total_sims)


def compute_signal(M: pd.DataFrame, strategy: str = "momentum") -> pd.DataFrame:
    if strategy == "momentum":
        M["signal"] = np.where(
            M["CLOSE_SMA_9D"] > M["CLOSE_SMA_21D"], 1, 0
        )  # long signal
        M["signal"] = np.where(
            M["CLOSE_SMA_9D"] < M["CLOSE_SMA_21D"], -1, M["signal"]
        )  # short signal
    return M


def pivotToClose(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        df["DATE"] = pd.to_datetime(df["DATE"])
    transformed_df = df.pivot(index="DATE", columns="ID", values="CLOSE")

    return transformed_df


def getData(
    df: pd.DataFrame, stocks: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    stockData = df[df["ID"].isin(stocks) & (df["DATE"] > start) & (df["DATE"] < end)]
    stockDataPivoted = stockData.pivot(index="DATE", columns="ID", values="CLOSE")
    log_returns = np.log(stockDataPivoted / stockDataPivoted.shift(1))
    log_returns = log_returns.dropna()
    meanReturns = log_returns.mean()
    covMatrix = log_returns.cov()
    return (log_returns, meanReturns, covMatrix)
