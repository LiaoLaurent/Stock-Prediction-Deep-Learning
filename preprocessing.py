import numpy as np
import pandas as pd
import argparse
import ta
import matplotlib.pyplot as plt


def load_data(data_path):
    df = pd.read_parquet(data_path)
    df = df.reset_index().set_index("ts_event")
    return df


def preprocess_data(df):
    df["mid_price"] = df[["ask_px_00", "bid_px_00"]].mean(axis=1)
    df["mid_price"] = df["mid_price"].combine_first(df["ask_px_00"])
    df["mid_price"] = df["mid_price"].combine_first(df["bid_px_00"])

    mid_prices = pd.DataFrame(
        {
            "mid_price_high": df["mid_price"].resample("s").max().ffill(),
            "mid_price_low": df["mid_price"].resample("s").min().ffill(),
            "mid_price_close": df["mid_price"].resample("s").last().ffill(),
            "mid_price_open": df["mid_price"].resample("s").first().ffill(),
        }
    )

    mid_prices["Returns"] = mid_prices["mid_price_close"].pct_change()
    mid_prices["Target"] = np.sign(mid_prices["Returns"])

    grouped = (
        df.groupby([pd.Grouper(freq="s"), "action", "side"])["size"].sum().reset_index()
    )
    order_sizes = grouped.pivot_table(
        index="ts_event", columns=["action", "side"], values="size", fill_value=0
    )
    order_sizes.drop(
        columns=[("A", "N"), ("C", "N"), ("T", "N")], inplace=True, errors="ignore"
    )

    action_mapping = {"A": "add", "C": "cancel", "T": "trade"}
    side_mapping = {"A": "ask", "B": "bid"}
    order_sizes.columns = [
        f"{action_mapping[action]}_{side_mapping[side]}_size"
        for action, side in order_sizes.columns
    ]
    order_sizes = order_sizes.reindex(mid_prices.index, fill_value=0)

    df_combined = pd.concat([mid_prices, order_sizes], axis=1)

    df_combined.dropna(inplace=True)

    hours, minutes, seconds = (
        df_combined.index.hour,
        df_combined.index.minute,
        df_combined.index.second,
    )
    time_features = pd.DataFrame(
        {
            "hour": hours,
            "sin_hour": np.sin(2 * np.pi * hours / 24),
            "cos_hour": np.cos(2 * np.pi * hours / 24),
            "minute": minutes,
            "sin_min": np.sin(2 * np.pi * minutes / 60),
            "cos_min": np.cos(2 * np.pi * minutes / 60),
            "second": seconds,
            "sin_sec": np.sin(2 * np.pi * seconds / 60),
            "cos_sec": np.cos(2 * np.pi * seconds / 60),
        },
        index=df_combined.index,
    )

    return pd.concat([df_combined, time_features], axis=1)


def compute_hft_indicators(df):
    indicators = df.copy()

    indicators["EMA_5"] = ta.trend.ema_indicator(
        indicators["mid_price_close"], window=5
    )
    indicators["MA_5"] = (
        indicators["mid_price_close"].rolling(window=5, min_periods=1).mean()
    )

    indicators["Bollinger_Upper"] = indicators["MA_5"] + (
        indicators["mid_price_close"].rolling(5).std() * 2
    )
    indicators["Bollinger_Lower"] = indicators["MA_5"] - (
        indicators["mid_price_close"].rolling(5).std() * 2
    )

    indicators["High_Shift"] = indicators["mid_price_high"].shift(1)
    indicators["Low_Shift"] = indicators["mid_price_low"].shift(1)

    indicators["DMP_3"] = (
        pd.Series(
            np.where(
                (
                    indicators["mid_price_high"] - indicators["High_Shift"]
                    > indicators["Low_Shift"] - indicators["mid_price_low"]
                ),
                np.maximum(indicators["mid_price_high"] - indicators["High_Shift"], 0),
                0,
            ),
            index=df.index,
        )
        .rolling(3, min_periods=1)
        .sum()
    )

    indicators["DMN_3"] = (
        pd.Series(
            np.where(
                (
                    indicators["Low_Shift"] - indicators["mid_price_low"]
                    > indicators["mid_price_high"] - indicators["High_Shift"]
                ),
                np.maximum(indicators["Low_Shift"] - indicators["mid_price_low"], 0),
                0,
            ),
            index=df.index,
        )
        .rolling(3, min_periods=1)
        .sum()
    )

    indicators["OLL3"] = (
        indicators["mid_price_open"]
        - indicators["mid_price_low"].rolling(3, min_periods=1).min()
    )
    indicators["OLL5"] = (
        indicators["mid_price_open"]
        - indicators["mid_price_low"].rolling(5, min_periods=1).min()
    )

    indicators["STOCHk_7_3_3"] = ta.momentum.stoch(
        indicators["mid_price_high"],
        indicators["mid_price_low"],
        indicators["mid_price_close"],
        window=7,
        smooth_window=3,
    )
    indicators["STOCHd_7_3_3"] = (
        indicators["STOCHk_7_3_3"].rolling(3, min_periods=1).mean()
    )

    indicators.drop(columns=["High_Shift", "Low_Shift"], inplace=True)

    return indicators.ffill().iloc[6:]
