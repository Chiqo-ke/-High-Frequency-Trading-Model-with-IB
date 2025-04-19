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
        # Validate input size
        if input_size <= 0:
            raise ValueError(f"Invalid input size: {input_size}")
            
        # Calculate hidden layer sizes based on input size
        h1_size = max(64, input_size * 2)
        h2_size = max(32, input_size)
        
        self.fc1 = nn.Linear(input_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, output_size)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(h1_size)
        self.batch_norm2 = nn.BatchNorm1d(h2_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Ensure input tensor is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.batch_size = RL_CONFIG['batch_size']
        
        self.gamma = RL_CONFIG['gamma']
        self.epsilon = RL_CONFIG['epsilon_start']
        self.epsilon_min = RL_CONFIG['epsilon_end']
        self.epsilon_decay = RL_CONFIG['epsilon_decay']
        
        # Create model with validated dimensions
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=RL_CONFIG['learning_rate'])
        
    def remember(self, state, action, reward, next_state, done):
        # Ensure states are numpy arrays
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Ensure state is a numpy array
        state = np.asarray(state, dtype=np.float32)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state)
        # Ensure state tensor has correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert experience to numpy arrays first
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Calculate target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
