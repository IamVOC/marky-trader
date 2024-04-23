
def format_df(df):
    # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
    df["feature_close"] = df["close"].pct_change()

    # Create the feature : open[t] / close[t]
    df["feature_open"] = df["open"]/df["close"]

    # Create the feature : high[t] / close[t]
    df["feature_high"] = df["high"]/df["close"]

    # Create the feature : low[t] / close[t]
    df["feature_low"] = df["low"]/df["close"]

     # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
    df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()

    df.dropna(inplace= True)

    return df