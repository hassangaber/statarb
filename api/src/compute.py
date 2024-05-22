import pandas as pd
import numpy as np

import pandas as pd


def dollar_bars(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Convert time bars to dollar bars.

    Parameters:
        data (pd.DataFrame): Original time bars data with columns ['DATE', 'CLOSE', 'VOLUME', ...].
        threshold (float): Dollar volume threshold for creating a new bar.

    Returns:
        pd.DataFrame: Dollar bars.
    """
    data["dollar_volume"] = data["CLOSE"] * data["VOLUME"]
    dollar_bars = []
    cum_dollar_volume = 0
    bar_start_idx = 0

    for i in range(len(data)):
        cum_dollar_volume += data.iloc[i]["dollar_volume"]

        if cum_dollar_volume >= threshold:
            bar_end_idx = i
            bar = data.iloc[bar_start_idx : bar_end_idx + 1]

            dollar_bars.append(
                {
                    "DATE": data.iloc[bar_end_idx]["DATE"],
                    "OPEN": bar.iloc[0]["CLOSE"],
                    "HIGH": bar["CLOSE"].max(),
                    "LOW": bar["CLOSE"].min(),
                    "CLOSE": bar.iloc[-1]["CLOSE"],
                    "VOLUME": bar["VOLUME"].sum(),
                    "dollar_volume": cum_dollar_volume,
                    # Add other necessary columns here if needed
                }
            )

            cum_dollar_volume = 0
            bar_start_idx = bar_end_idx + 1

    dollar_bars_df = pd.DataFrame(dollar_bars)
    return dollar_bars_df


def compute_signal(M: pd.DataFrame, strategy: str = "momentum") -> pd.DataFrame:
    if strategy == "momentum":
        M["signal"] = np.where(M["CLOSE_SMA_9D"] > M["CLOSE_SMA_21D"], 1, 0)  # long signal
        M["signal"] = np.where(M["CLOSE_SMA_9D"] < M["CLOSE_SMA_21D"], -1, M["signal"])  # short signal
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
