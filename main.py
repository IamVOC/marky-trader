from custom_downloader import download
import datetime
import pandas as pd

import gymnasium as gym
import gym_trading_env
import numpy as np


for i in range(5):
    download(exchange_names = ["binance"], 
            symbols=["BTC/USDT"], 
            timeframe="1h", 
            dir="data",
            since=datetime.datetime(year=2018 + i, month=1, day=1),
            until=datetime.datetime(year=2019 + i, month=1, day=1)
    )


def preprocess(df : pd.DataFrame):

    # Create your features
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    df.dropna(inplace= True)
    return df

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


env = gym.make("MultiDatasetTradingEnv",
        name= "BTCUSD",
        dataset_dir = "./data/*.pkl", # Your dataset with your custom features
        preprocess=preprocess,
        dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
        positions = [ -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.001/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 10000,
    )

env.unwrapped.add_metric('Position changes', lambda history: np.sum(np.diff(history['position']) != 0))
env.unwrapped.add_metric('Episode lenght', lambda history: len(history['position']))
 
done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    observation, reward, done, truncated, info = env.step(position_index)

env.unwrapped.save_for_render(dir="logs")