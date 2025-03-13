import pandas as pd
import ta
import numpy as np

market_open = pd.Timestamp("09:30:00").time()
market_close = pd.Timestamp("16:00:00").time()


def compute_order_book_features(
    raw_data, resample_rate, mid_price_variation_class_threshold=0.0
):

    df = raw_data.copy()

    df["mid_price"] = df[["ask_px_00", "bid_px_00"]].mean(axis=1)
    df["mid_price"] = df["mid_price"].fillna(df["ask_px_00"])
    df["mid_price"] = df["mid_price"].fillna(df["bid_px_00"])

    df["bid_ask_spread"] = df["ask_px_00"] - df["bid_px_00"]

    df["weighted_mid_price"] = (
        df["ask_px_00"] * df["bid_sz_00"] + df["bid_px_00"] * df["ask_sz_00"]
    ) / (df["bid_sz_00"] + df["ask_sz_00"])
    df["weighted_mid_price"] = df["weighted_mid_price"].fillna(df["ask_px_00"])
    df["weighted_mid_price"] = df["weighted_mid_price"].fillna(df["bid_px_00"])

    trade_mask = df["action"] == "T"
    add_mask = df["action"] == "A"
    cancel_mask = df["action"] == "C"
    bid_mask = df["side"] == "B"
    ask_mask = df["side"] == "A"

    # Aggregate resampled data
    order_book_data = df.resample(resample_rate).agg(
        {
            "mid_price": ["first", "last", "max", "min", "mean", "std"],
            "weighted_mid_price": ["first", "last", "mean"],
            "bid_ask_spread": ["last", "mean", "std"],
            "bid_px_00": ["last", "mean"],
            "ask_px_00": ["last", "mean"],
            "bid_sz_00": ["last", "mean", "std"],
            "ask_sz_00": ["last", "mean", "std"],
            "ask_px_01": "mean",
            "bid_px_01": "mean",
        }
    )

    order_book_data.columns = [
        "mid_price_first",  # mid price
        "mid_price_last",
        "mid_price_high",
        "mid_price_low",
        "mid_price_mean",
        "std_mid_price",
        "weighted_mid_price_first",  # weighted mid price
        "weighted_mid_price_last",
        "weighted_mid_price_mean",
        "last_spread",  # bid-ask spread
        "mean_spread",
        "std_spread",
        "last_best_bid_price",  # bid price
        "mean_best_bid_price",
        "last_best_ask_price",  # ask price
        "mean_best_ask_price",
        "last_best_bid_size",  # bid size
        "mean_best_bid_size",
        "std_best_bid_size",
        "last_best_ask_size",  # ask size
        "mean_best_ask_size",
        "std_best_ask_size",
        "mean_second_ask_price",  # mean second ask price
        "mean_second_bid_price",  # mean second bid price
    ]

    # When there are no orders for the duration of the sample period, fill with the last available value
    order_book_data[
        [
            "last_best_bid_size",
            "last_best_ask_size",
            "mean_best_bid_size",
            "mean_best_ask_size",
        ]
    ] = order_book_data[
        [
            "last_best_bid_size",
            "last_best_ask_size",
            "mean_best_bid_size",
            "mean_best_ask_size",
        ]
    ].ffill()
    order_book_data[
        [
            "last_best_bid_price",
            "last_best_ask_price",
            "mean_best_bid_size",
            "mean_best_ask_size",
        ]
    ] = order_book_data[
        [
            "last_best_bid_price",
            "last_best_ask_price",
            "mean_best_bid_size",
            "mean_best_ask_size",
        ]
    ].ffill()

    # POSSIBLE TARGETS
    order_book_data["mid_price_variation"] = (
        order_book_data["mid_price_last"] / order_book_data["mid_price_first"] - 1
    )
    # Classify mid-price variations
    order_book_data["mid_price_variation_class"] = order_book_data[
        "mid_price_variation_class"
    ] = (
        np.where(
            np.abs(order_book_data["mid_price_variation"])
            > mid_price_variation_class_threshold,
            np.sign(order_book_data["mid_price_variation"]),
            0,
        )
        + 1
    )
    order_book_data["mean_mid_price_variation"] = (
        order_book_data["mid_price_mean"] / order_book_data["mid_price_first"] - 1
    )
    # Classify mean mid-price variations
    order_book_data["mean_mid_price_variation_class"] = (
        order_book_data["mean_mid_price_variation"] >= 0
    ).astype(int)
    order_book_data["next_5_mean_mid_price_variation_class"] = (
        order_book_data["mean_mid_price_variation"].shift(-4).rolling(window=5).mean()
        > 0
    ).astype(int)

    order_book_data["weighted_mid_price_variation"] = (
        order_book_data["weighted_mid_price_last"]
        / order_book_data["weighted_mid_price_first"]
        - 1
    )
    # Classify weighted mid-price variations
    order_book_data["weighted_mid_price_variation_class"] = (
        order_book_data["weighted_mid_price_variation"] > 0
    ).astype(int)

    # Compute trade-related values
    trade_prices = (
        df.loc[trade_mask, "price"]
        .resample(resample_rate)
        .agg(["first", "last", "max", "min"])
    )
    trade_prices.columns = ["trade_open", "trade_close", "trade_high", "trade_low"]

    # Compute total bid/ask volume (additions - cancellations)
    order_book_data["total_bid_volume"] = df.loc[add_mask & bid_mask, "size"].astype(
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
    order_book_data["total_ask_volume"] = df.loc[add_mask & ask_mask, "size"].astype(
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
    order_book_data["mean_order_book_imbalance"] = (
        order_book_data["mean_best_bid_size"].astype("int64")
        - order_book_data["mean_best_ask_size"].astype("int64")
    ) / (order_book_data["mean_best_bid_size"] + order_book_data["mean_best_ask_size"])

    order_book_data["last_order_book_imbalance"] = (
        order_book_data["last_best_bid_size"].astype("int64")
        - order_book_data["last_best_ask_size"].astype("int64")
    ) / (order_book_data["last_best_bid_size"] + order_book_data["last_best_ask_size"])

    order_book_data["total_net_order_flow"] = order_book_data[
        "last_best_bid_size"
    ].astype("int64") - order_book_data["last_best_ask_size"].astype("int64")

    # Count order actions
    order_book_data["num_added_orders"] = (
        df.loc[add_mask].resample(resample_rate).size()
    )
    order_book_data["num_canceled_orders"] = (
        df.loc[cancel_mask].resample(resample_rate).size()
    )
    order_book_data["num_traded_orders"] = (
        df.loc[trade_mask].resample(resample_rate).size()
    )

    # Compute rolling averages (5s window)
    order_book_data["order_book_imbalance_5s"] = (
        order_book_data["mean_order_book_imbalance"].rolling(5).mean()
    )

    order_book_data["order_flow_5s"] = (
        order_book_data["total_net_order_flow"].rolling(5).mean()
    )

    # Compute bid/ask price/volume variations
    order_book_data["bid_volume_variation"] = (
        order_book_data["last_best_bid_size"].astype("int64").diff()
    )
    order_book_data["ask_volume_variation"] = (
        order_book_data["last_best_ask_size"].astype("int64").diff()
    )
    order_book_data["bid_price_variation"] = (
        order_book_data["last_best_bid_price"].astype("int64").diff()
    )
    order_book_data["ask_price_variation"] = (
        order_book_data["last_best_ask_price"].astype("int64").diff()
    )

    order_book_data = order_book_data.merge(
        trade_prices, left_index=True, right_index=True, how="left"
    )

    order_book_data.dropna(inplace=True)

    return order_book_data


def add_technical_indicators(df):

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

    df["STOCHk_7_3_3"] = df["STOCHk_7_3_3"].ffill()
    df["STOCHd_7_3_3"] = df["STOCHd_7_3_3"].ffill()

    df["MACD_8_21_5"] = ta.trend.MACD(
        df["mid_price_mean"], window_slow=21, window_fast=8, window_sign=5
    ).macd_diff()
    df["RSI_7"] = ta.momentum.RSIIndicator(df["mid_price_mean"], window=7).rsi()
    df["AO_5_10"] = ta.momentum.AwesomeOscillatorIndicator(
        df["mid_price_high"], df["mid_price_low"], window1=5, window2=10
    ).awesome_oscillator()

    ### MOVING AVERAGES ###
    df["EMA_15"] = ta.trend.EMAIndicator(
        df["mid_price_mean"], window=15
    ).ema_indicator()
    df["HMA_10"] = ta.trend.WMAIndicator(df["mid_price_mean"], window=10).wma()
    df["KAMA_3_2_10"] = ta.momentum.KAMAIndicator(
        df["mid_price_mean"], window=3, pow1=2, pow2=10
    ).kama()
    df["MA_10"] = ta.trend.SMAIndicator(df["mid_price_mean"], window=10).sma_indicator()
    df["MA_20"] = ta.trend.SMAIndicator(df["mid_price_mean"], window=20).sma_indicator()

    # Rolling CO (Last - First)
    for w in [3, 4, 5, 6]:
        df[f"rmCO({w})"] = (
            (df["mid_price_last"] - df["mid_price_first"]).rolling(window=w).mean()
        )

    ### VOLATILITY INDICATORS ###
    df["Bollinger_Upper"] = ta.volatility.BollingerBands(
        df["mid_price_mean"], window=20, window_dev=2
    ).bollinger_hband()
    df["Bollinger_Lower"] = ta.volatility.BollingerBands(
        df["mid_price_mean"], window=20, window_dev=2
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
    if not isinstance(combined_data.index, pd.DatetimeIndex):
        combined_data.index = pd.to_datetime(combined_data.index)

    market_open_time = combined_data.index.normalize() + pd.Timedelta(
        hours=9, minutes=30
    )
    combined_data["time_since_open"] = (
        combined_data.index - market_open_time
    ).total_seconds()

    combined_data["is_monday"] = (combined_data.index.weekday == 0).astype(int)
    combined_data["is_tuesday"] = (combined_data.index.weekday == 1).astype(int)
    combined_data["is_wednesday"] = (combined_data.index.weekday == 2).astype(int)
    combined_data["is_thursday"] = (combined_data.index.weekday == 3).astype(int)
    combined_data["is_friday"] = (combined_data.index.weekday == 4).astype(int)

    return combined_data


def process_and_combine_data(
    start_date,
    end_date,
    mid_price_variation_class_threshold=0.0,
    data_folder="../AAPL_data",
    sampling_rate="1s",
):

    # Convert dates to datetime format
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Generate business days within the date range
    trading_days = pd.bdate_range(start=start_dt, end=end_dt)

    file_paths = [
        f"{data_folder}/AAPL_{date.strftime('%Y-%m-%d')}_xnas-itch.parquet"
        for date in trading_days
    ]

    daily_data_list = []

    for file_path in file_paths:
        raw_data = pd.read_parquet(file_path).between_time(market_open, market_close)

        order_book_data = compute_order_book_features(
            raw_data, sampling_rate, mid_price_variation_class_threshold
        )

        enriched_data = add_technical_indicators(order_book_data)

        daily_data_list.append(enriched_data)

    combined_data = pd.concat(daily_data_list)

    return add_time_features(combined_data)
