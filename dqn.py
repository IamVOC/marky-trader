import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gymnasium as gym
import gym_trading_env
import random
import numpy as np
import matplotlib.pyplot as plt
from reader import get_df
from metrics import confidence_interval
from scipy.stats import gaussian_kde


def reward_function(history):
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

positions = [-1, 0, 1]

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = get_df('1h-data/binance-BTCUSDT-1h.pkl'),
        positions = positions,
        initial_position = 'random',
        max_episode_duration = 24*730,
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        portfolio_initial_value = 1000,
        reward_function = reward_function,
        windows = 24*7,
    )

observation_space = 2184
action_space = 3

EPISODES = 100000
LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1
EXPLORATION_DECAY = 0.999998
EXPLORATION_MIN = 0.01

FC1_DIMS = 1024
FC2_DIMS = 256
DEVICE = torch.device("cuda")

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(observation_space, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, observation_space),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, observation_space),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        mem_max = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(mem_max, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN:
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network()

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return random.randrange(3)
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + GAMMA * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate

agent = DQN()

total_rewards = []
for i in range(1, EPISODES):
    state = env.reset()
    state = state[0]
    score = 0
    rewards = []
    print(agent.exploration_rate)
    while True:
        state = state.flatten()
        action = agent.choose_action(state)
        state_, reward, done, truncated, info = env.step(action)
        state_ = np.reshape(state_, [1, observation_space])
        agent.memory.add(state, action, reward, state_, done)
        agent.learn()
        state = state_
        score += reward
        rewards.append(reward)

        if done or truncated:
            break
            
    if (i + 1) % 10 == 0:
        torch.save(agent.network.state_dict(), f'models/dqn/dqn_{i}.pt')
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(list(range(i-1)), total_rewards)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Total reward')
        fig.savefig(f'diagrams/reward.png')
        plt.close(fig)
        with open('diagrams/statistic_{episode}.txt', '+w') as f:
            f.write(f'{repr(confidence_interval(total_rewards, 0.95))}\n')
            f.write(f'{repr(confidence_interval(total_rewards, 0.90))}\n')
            f.write(f'{repr(np.mean(total_rewards))}\n')
    total_rewards.append(sum(rewards))
