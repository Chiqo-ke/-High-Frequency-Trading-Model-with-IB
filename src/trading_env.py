import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict
import pandas as pd

class ForexTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000, state_dim: int = None):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = None
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        
        # Dynamically set observation space dimension
        if state_dim is None:
            state_dim = len(df.columns) - 1  # Exclude 'Time' column
            
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,),
            dtype=np.float32
        )

    def _get_state(self) -> np.array:
        """Create state vector from current market data and position"""
        current_data = self.df.iloc[self.current_step]
        
        # Get all numeric columns except 'Time'
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        technical_features = current_data[numeric_columns].values
        
        position_flag = 0 if self.position is None else (1 if self.position == 'long' else -1)
        
        state = np.append([self.balance, position_flag], technical_features)
        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict]:
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        
        # Execute trading action
        if action == 0:  # Buy
            if self.position is None:
                self.position = 'long'
                reward = -current_price * 0.0001  # Trading fee
        elif action == 1:  # Sell
            if self.position is None:
                self.position = 'short'
                reward = -current_price * 0.0001  # Trading fee
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate reward
        if not done and self.position is not None:
            next_price = self.df.iloc[self.current_step]['Close']
            price_change = (next_price - current_price) / current_price
            reward += price_change if self.position == 'long' else -price_change
        
        return self._get_state(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        return self._get_state()
