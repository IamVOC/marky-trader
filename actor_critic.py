import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from metrics import confidence_interval
from scipy.stats import gaussian_kde
import gymnasium as gym
from reader import get_df

device = 'cuda'

class NNAC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNAC, self).__init__()
        self.fc_actor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, output_dim),
        )
        self.double()

    def forward(self, x):
        x = self.fc_actor(x)
        return x

def train(actor, critic, env, num_episodes=100000):
    actor_optimizer = optim.Adam(actor.parameters())
    critic_optimizer = optim.Adam(critic.parameters())
    total_rewards = []
    actions = []
    for episode in range(num_episodes):
        print(f'Episode {episode+1} ============================')
        state = env.reset()
        state = state[0]
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        done = False
        truncated = False
        state = state.flatten()
        while not done and not truncated:
            state = torch.tensor(state).double().cuda()
            logits, value = actor(state), critic(state)
            dist = Categorical(F.sigmoid(logits))

            action = dist.sample()
            actions.append(action)
            next_state, reward, done, truncated, _ = env.step(action.cpu())
            next_state = next_state.flatten()

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

        next_state = torch.tensor(next_state).double().cuda()
        next_value = critic(next_state)
        R = next_value
        rtgs = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + 0.9 * R * masks[step]
            rtgs.insert(0, R)

        log_probs = torch.cat(log_probs)
        rtgs = torch.cat(rtgs).detach()
        values = torch.cat(values)

        advantage = rtgs - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

        total_rewards.append(sum(rewards).cpu().item())
        
        if (episode + 1) % 50 == 0:
            res = gaussian_kde(total_rewards)
            x = np.linspace(min(total_rewards), max(total_rewards), 12)
            y = res.evaluate(x)
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(x, y)
            fig.savefig(f'diagrams/reward_distribution_episode_{episode}.png')
            plt.close(fig)
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.hist([c.cpu() for c in actions], bins=3, edgecolor="white")
            fig.savefig(f'diagrams/action_distribution_episode_{episode}.png')
            plt.close(fig)
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(list(range(episode+1)), total_rewards)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Total reward')
            fig.savefig(f'diagrams/reward_episode_{episode}.png')
            plt.close(fig)
            with open('diagrams/statistic_{episode}.txt', '+w') as f:
                f.write(f'{repr(confidence_interval(total_rewards, 0.95))}\n')
                f.write(f'{repr(confidence_interval(total_rewards, 0.90))}\n')
                f.write(f'{repr(np.mean(total_rewards))}\n')
            torch.save(actor.state_dict(), f'models/actor_critic/actor_very_long_{episode}.pt')

device = ("cuda"
          if torch.cuda.is_available()
          else "cpu")

def reward_function(history):
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

positions = [-1, 0, 1]

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = get_df('data/binance-BTCUSDT-1h.pkl'),
        positions = positions,
        initial_position = 'random',
        max_episode_duration = 24 * 730,
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        portfolio_initial_value = 1000,
        reward_function = reward_function,
        windows = 24*7,
    )


actor = NNAC(2184, len(positions)).to(device)
critic = NNAC(2184, 1).to(device)
train(actor, critic, env)
            