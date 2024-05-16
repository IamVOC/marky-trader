import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.stats



def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, env.action_space.n)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

def train(env, agent, lr=0.01, gamma=0.97):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    total_returns = []
    for episode in range(200):
        state = env.reset()
        state = state[0]
        episode_return = 0
        episode_log_probs = []
        done, truncated = False, False
        while not done and not truncated:
            state = torch.FloatTensor(state)
            action_probs = agent.forward(state)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, truncated, _ = env.step(action)
            episode_return += reward
            episode_log_probs.append(torch.log(action_probs[action]))
            state = next_state
        
        total_returns.append(episode_return)
        
        J = np.repeat(episode_return, len(episode_log_probs))
        policy_gradient = []
        for log_prob in reversed(episode_log_probs):  
            policy_gradient.append(-log_prob * torch.tensor(J))
            J = J * gamma 
        
        actor_loss = torch.stack(policy_gradient).sum()
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            env.unwrapped.save_for_render(dir="logs")
            torch.save(agent.state_dict(), f'models/reinforce/r_{episode}.pt')
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(list(range(episode+1)), total_returns)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Total reward')
            fig.savefig(f'diagrams/reinforce/reward.png')
            plt.close(fig)
            with open(f'diagrams/reinforce/statstic_{episode}.txt', '+w') as f:
                f.write(f'{repr(confidence_interval(total_returns, 0.95))}\n')
                f.write(f'{repr(confidence_interval(total_returns, 0.90))}\n')
                f.write(f'{repr(np.mean(episode_return))}\n')
