import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from config import RL_CONFIG

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=RL_CONFIG['memory_size'])
        self.batch_size = RL_CONFIG['batch_size']
        
        self.gamma = RL_CONFIG['gamma']
        self.epsilon = RL_CONFIG['epsilon_start']
        self.epsilon_min = RL_CONFIG['epsilon_end']
        self.epsilon_decay = RL_CONFIG['epsilon_decay']
        
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=RL_CONFIG['learning_rate'])
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([i[0] for i in batch])
        actions = torch.LongTensor([i[1] for i in batch])
        rewards = torch.FloatTensor([i[2] for i in batch])
        next_states = torch.FloatTensor([i[3] for i in batch])
        dones = torch.FloatTensor([i[4] for i in batch])
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
