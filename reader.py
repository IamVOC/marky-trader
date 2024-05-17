import pandas as pd
import numpy as np

from dynamic_data import calc_lower_bollinger_bands, calc_upper_bollinger_bands, calc_stochastic_fast, calc_stochastic_slow, calc_adx, calc_rsi


def get_df(name: str):

    df = pd.read_pickle(name)
    print(df)
    return preprocess(df)

def preprocess(df: pd.DataFrame):

    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(24*7).max()
    df["feature_low_bol"] = calc_lower_bollinger_bands(df)
    df["feature_up_bol"] = calc_upper_bollinger_bands(df)
    df["feature_stoch_slow"] = calc_stochastic_fast(df)
    df["feature_stoch_fast"] = calc_stochastic_slow(df)
    df["feature_adx"] = calc_adx(df)
    df["feature_rsi"] = calc_rsi(df)

    df.dropna(inplace= True)
    return df
