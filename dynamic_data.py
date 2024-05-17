import pandas as pd

def calc_upper_bollinger_bands(df):
    boll_upper = df['close'].rolling(200).mean() + 2 * df['close'].rolling(200).std()
    return boll_upper

def calc_lower_bollinger_bands(df):
    boll_lower = df['close'].rolling(200).mean() - 2 * df['close'].rolling(200).std()
    return boll_lower

def calc_stochastic_fast(df):
    stoch_fast = 100 * (df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min())
    return stoch_fast

def calc_stochastic_slow(df):
    stoch_fast = 100 * (df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min())
    stoch_slow = stoch_fast.rolling(3).mean()
    return stoch_slow

def calc_adx(df):
    adx = abs(df['high'] - df['low']) / (df['high'] + df['low']) * 100
    adx = adx.ewm(com=14, adjust=False).mean()
    return adx

def calc_rsi(df):
    delta = df['close'].diff().abs()
    gain = df['close'].diff(1).clip(lower=0)
    loss = -df['close'].diff(1).clip(upper=0)
    rs = gain.ewm(com=14, adjust=False).mean() / loss.ewm(com=14, adjust=False).mean()
    rsi = 100 - 100 / (1 + rs)
    return rsi
