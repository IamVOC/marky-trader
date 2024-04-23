import gymnasium as gym
import gym_trading_env
import pandas as pd
import matplotlib.pyplot as plt

from epsilon import Agent
from foramt_df import format_df
from utils import print_metrics

df = pd.read_csv('data/trade1.csv')

df = format_df(df)

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df,
        positions = [-1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )


agent = Agent(epsilon=0.1, lr=0.8, discount=0.9)

num_episode = 30
total_reward = []

for episode in range(num_episode):
    episode_reward = 0
    state = env.reset()
    state = tuple(state[0])
    done, truncated = False, False
    episode_reward = 0
    while not done and not truncated:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = tuple(next_state)
        agent.update_Q(state, action, reward, next_state)
        state = next_state
        episode_reward += reward
    total_reward.append(episode_reward)
    print(f'==========================={episode+1}===========================')


print_metrics(total_reward=total_reward)

plt.plot([i for i in range(len(total_reward))], total_reward)    
plt.show()