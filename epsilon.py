import random
import numpy as np

class Agent():

    def __init__(self, epsilon, lr, discount):
        self.epsilon = epsilon
        self.lr = lr
        self.Q_values = {}
        self.discount = discount

    def choose_action(self, state):
        if state not in self.Q_values:
            self.Q_values[state] = [-1, 0, 1]

        if random.random() < self.epsilon:
            return random.choice((1, 0, -1))
        else:
            return np.argmax(self.Q_values[state]) 
        
    def update_Q(self, state, action, reward, next_state):
        if next_state not in self.Q_values:
            self.Q_values[next_state] = [-1, 0, 1]

        max_Q_next = max(self.Q_values[next_state])
        self.Q_values[state][action] = self.Q_values[state][action] + self.lr * (reward + self.discount * max_Q_next - self.Q_values[state][action])


    