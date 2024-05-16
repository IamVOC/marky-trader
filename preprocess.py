import pandas as pd

def preprocess(df : pd.DataFrame):

    # Create your features
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    # df["sma10"] = df["close"].rolling(10).mean()
    df.dropna(inplace= True)
    return df
