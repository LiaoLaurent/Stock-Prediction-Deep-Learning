import pandas as pd
import ta
import numpy as np

market_open = pd.Timestamp("09:30:00").time()
market_close = pd.Timestamp("16:00:00").time()


def compute_order_book_features(raw_data, resample_rate):

    df = raw_data.copy()

    # Compute the mid-price as the average of best bid/ask prices
    df["mid_price"] = df[["ask_px_00", "bid_px_00"]].mean(axis=1)
    df["mid_price"] = df["mid_price"].fillna(df["ask_px_00"])
    df["mid_price"] = df["mid_price"].fillna(df["bid_px_00"])

    # Precompute masks for filtering specific actions
    trade_mask = df["action"] == "T"
    add_mask = df["action"] == "A"
    cancel_mask = df["action"] == "C"
    bid_mask = df["side"] == "B"
    ask_mask = df["side"] == "A"

    # Aggregate resampled data using efficient functions
    resampled_data = df.resample(resample_rate).agg(
        {
            "mid_price": ["first", "last", "max", "min", "mean", "std"],
            "bid_px_00": ["last", "mean", "std"],
            "ask_px_00": ["last", "mean", "std"],
            "bid_sz_00": ["last", "mean", "std"],
            "ask_sz_00": ["last", "mean", "std"],
            "ask_px_01": "mean",
            "bid_px_01": "mean",
        }
    )

    # Rename columns for clarity
    resampled_data.columns = [
        "mid_price_first",
        "mid_price_last",
        "mid_price_high",
        "mid_price_low",
        "mean_mid_price",
        "std_mid_price",
        "best_bid_price",
        "mean_best_bid_price",
        "std_best_bid_price",
        "best_ask_price",
        "mean_best_ask_price",
        "std_best_ask_price",
        "best_bid_size",
        "mean_best_bid_size",
        "std_best_bid_size",
        "best_ask_size",
        "mean_best_ask_size",
        "std_best_ask_size",
        "mean_second_bid_ask_spread",
        "mean_second_bid_price",
    ]

    # Forward-fill missing bid/ask sizes
    resampled_data[["best_bid_size", "best_ask_size"]] = resampled_data[
        ["best_bid_size", "best_ask_size"]
    ].ffill()

    # Compute derived features
    resampled_data["bid_ask_spread"] = (
        resampled_data["best_ask_price"] - resampled_data["best_bid_price"]
    )
    resampled_data["mid_price_variation"] = (
        resampled_data["mid_price_last"] / resampled_data["mid_price_first"] - 1
    )
    resampled_data["mid_price_variation_class"] = (
        np.sign(resampled_data["mid_price_variation"]) + 1
    )

    # Compute trade-related values
    trade_prices = (
        df.loc[trade_mask, "price"]
        .resample(resample_rate)
        .agg(["first", "last", "max", "min"])
    )
    trade_prices.columns = ["trade_open", "trade_close", "trade_high", "trade_low"]

    # Compute total bid/ask volume (additions - cancellations)
    resampled_data["total_bid_volume"] = df.loc[add_mask & bid_mask, "size"].astype(
        "int64"
    ).resample(resample_rate).sum().fillna(0) - df.loc[
        cancel_mask & bid_mask, "size"
    ].astype(
        "int64"
    ).resample(
        resample_rate
    ).sum().fillna(
        0
    )
    resampled_data["total_ask_volume"] = df.loc[add_mask & ask_mask, "size"].astype(
        "int64"
    ).resample(resample_rate).sum().fillna(0) - df.loc[
        cancel_mask & ask_mask, "size"
    ].astype(
        "int64"
    ).resample(
        resample_rate
    ).sum().fillna(
        0
    )

    # Compute order book imbalance metrics
    resampled_data["mean_order_book_imbalance"] = (
        resampled_data["mean_best_bid_size"] - resampled_data["mean_best_ask_size"]
    )
    resampled_data["mean_volume_ratio_bid_ask"] = (
        resampled_data["mean_best_bid_size"] / resampled_data["mean_best_ask_size"]
    )
    resampled_data["total_net_order_flow"] = resampled_data["best_bid_size"].astype(
        "int64"
    ) - resampled_data["best_ask_size"].astype("int64")

    # Count order actions
    resampled_data["num_added_orders"] = df.loc[add_mask].resample(resample_rate).size()
    resampled_data["num_canceled_orders"] = (
        df.loc[cancel_mask].resample(resample_rate).size()
    )
    resampled_data["num_traded_orders"] = (
        df.loc[trade_mask].resample(resample_rate).size()
    )

    # Compute rolling averages (5s window)
    resampled_data["order_book_imbalance_5s"] = (
        resampled_data["mean_order_book_imbalance"].rolling(5).mean()
    )
    resampled_data["volume_ratio_5s"] = (
        resampled_data["mean_volume_ratio_bid_ask"].rolling(5).mean()
    )
    resampled_data["order_flow_5s"] = (
        resampled_data["total_net_order_flow"].rolling(5).mean()
    )

    # Merge trade data with resampled data
    resampled_data = resampled_data.merge(
        trade_prices, left_index=True, right_index=True, how="left"
    )

    # Drop NaNs from rolling calculations
    resampled_data.dropna(inplace=True)

    return resampled_data


def add_technical_indicators(df):
    """
    Compute technical indicators for a given DataFrame with OHLC-like structure.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    ### TREND INDICATORS ###
    df["ADX_5"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=5
    ).adx()
    df["ADX_7"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=7
    ).adx()
    df["ADX_10"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=10
    ).adx()
    df["DMP_5"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=5
    ).adx_pos()
    df["DMP_10"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=10
    ).adx_pos()
    df["DMN_5"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=5
    ).adx_neg()
    df["DMN_10"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_last"], window=10
    ).adx_neg()

    # Aroon Indicator requires both high and low prices
    df["AROONU_7"] = ta.trend.AroonIndicator(
        df["mid_price_high"], df["mid_price_low"], window=7
    ).aroon_up()
    df["AROOND_7"] = ta.trend.AroonIndicator(
        df["mid_price_high"], df["mid_price_low"], window=7
    ).aroon_down()

    # Open-Low and Open-High calculations
    df["OLL3"] = df["mid_price_first"] - df["mid_price_low"].rolling(window=3).min()
    df["OLL5"] = df["mid_price_first"] - df["mid_price_low"].rolling(window=5).min()
    df["OLL10"] = df["mid_price_first"] - df["mid_price_low"].rolling(window=10).min()
    df["OLL15"] = df["mid_price_first"] - df["mid_price_low"].rolling(window=15).min()
    df["OHH3"] = df["mid_price_high"].rolling(window=3).max() - df["mid_price_first"]
    df["OHH5"] = df["mid_price_high"].rolling(window=5).max() - df["mid_price_first"]

    ### OSCILLATORS ###
    df["STOCHk_7_3_3"] = ta.momentum.StochasticOscillator(
        df["mid_price_high"],
        df["mid_price_low"],
        df["mid_price_last"],
        window=7,
        smooth_window=3,
    ).stoch()
    df["STOCHd_7_3_3"] = ta.momentum.StochasticOscillator(
        df["mid_price_high"],
        df["mid_price_low"],
        df["mid_price_last"],
        window=7,
        smooth_window=3,
    ).stoch_signal()

    # Avoid NaN Stochastic values
    df["STOCHk_7_3_3"] = df["STOCHk_7_3_3"].ffill()
    df["STOCHd_7_3_3"] = df["STOCHd_7_3_3"].ffill()

    df["MACD_8_21_5"] = ta.trend.MACD(
        df["mid_price_last"], window_slow=21, window_fast=8, window_sign=5
    ).macd_diff()
    df["RSI_7"] = ta.momentum.RSIIndicator(df["mid_price_last"], window=7).rsi()
    df["AO_5_10"] = ta.momentum.AwesomeOscillatorIndicator(
        df["mid_price_high"], df["mid_price_low"], window1=5, window2=10
    ).awesome_oscillator()

    ### MOVING AVERAGES ###
    df["EMA_15"] = ta.trend.EMAIndicator(
        df["mid_price_last"], window=15
    ).ema_indicator()
    df["HMA_10"] = ta.trend.WMAIndicator(
        df["mid_price_last"], window=10
    ).wma()  # HMA is not directly available in 'ta', using WMA as a placeholder
    df["KAMA_3_2_10"] = ta.momentum.KAMAIndicator(
        df["mid_price_last"], window=3, pow1=2, pow2=10
    ).kama()
    df["MA_10"] = ta.trend.SMAIndicator(df["mid_price_last"], window=10).sma_indicator()
    df["MA_20"] = ta.trend.SMAIndicator(df["mid_price_last"], window=20).sma_indicator()

    # Rolling CO (Last - First)
    for w in [3, 4, 5, 6]:
        df[f"rmCO({w})"] = (
            (df["mid_price_last"] - df["mid_price_first"]).rolling(window=w).mean()
        )

    ### VOLATILITY INDICATORS ###
    df["Bollinger_Upper"] = ta.volatility.BollingerBands(
        df["mid_price_last"], window=20, window_dev=2
    ).bollinger_hband()
    df["Bollinger_Lower"] = ta.volatility.BollingerBands(
        df["mid_price_last"], window=20, window_dev=2
    ).bollinger_lband()
    df["U_minus_L"] = df["Bollinger_Upper"] - df["Bollinger_Lower"]
    df["MA20dSTD"] = df["mid_price_last"].rolling(window=20).std()

    ### OTHER INDICATORS ###
    df["CO"] = df["mid_price_last"] - df["mid_price_first"]
    df["C1O1"] = df["CO"].shift(1)
    df["C2O2"] = df["CO"].shift(2)
    df["C3O3"] = df["CO"].shift(3)
    df["range"] = df["mid_price_high"] - df["mid_price_low"]
    df["OH1"] = df["mid_price_high"].shift(1) - df["mid_price_first"].shift(1)

    return df.dropna()


def add_time_features(combined_data):
    # Ensure the index is a DatetimeIndex
    if not isinstance(combined_data.index, pd.DatetimeIndex):
        combined_data.index = pd.to_datetime(combined_data.index)

    # Compute seconds since market open (9:30 AM)
    market_open_time = combined_data.index.normalize() + pd.Timedelta(
        hours=9, minutes=30
    )
    combined_data["time_since_open"] = (
        combined_data.index - market_open_time
    ).total_seconds()

    # Add binary features for Monday and Friday
    combined_data["is_monday"] = (combined_data.index.weekday == 0).astype(
        int
    )  # Monday = 0
    combined_data["is_friday"] = (combined_data.index.weekday == 4).astype(
        int
    )  # Friday = 4

    return combined_data


def process_and_combine_data(
    start_date, end_date, data_folder="../AAPL_data", sampling_rate="1s"
):
    """
    Processes and combines order book data for a given date range.

    Parameters:
    - start_date (str or datetime): Start date for data processing.
    - end_date (str or datetime): End date for data processing.
    - data_folder (str): Path to the folder containing parquet files.
    - sampling_rate (str): Resampling rate for computing order book features.

    Returns:
    - DataFrame: Combined and processed data with order book features and technical indicators.
    """
    # Convert dates to datetime format
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate business days within the date range
    trading_days = pd.bdate_range(start=start_dt, end=end_dt)

    # Construct file paths for each trading day
    file_paths = [
        f"{data_folder}/AAPL_{date.strftime('%Y-%m-%d')}_xnas-itch.parquet"
        for date in trading_days
    ]

    # List to store processed data for each day
    daily_data_list = []

    # Process each day's data
    for file_path in file_paths:
        # Load raw data and filter by market hours
        raw_data = pd.read_parquet(file_path).between_time(market_open, market_close)

        # Compute order book features
        order_book_data = compute_order_book_features(raw_data, sampling_rate)

        # Add technical indicators
        enriched_data = add_technical_indicators(order_book_data)

        # Store processed data
        daily_data_list.append(enriched_data)

    # Concatenate all processed daily data
    combined_data = pd.concat(daily_data_list)

    # Add time-based features and return final dataset
    return add_time_features(combined_data)
