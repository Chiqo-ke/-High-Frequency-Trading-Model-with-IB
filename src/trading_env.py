import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict
import pandas as pd

class ForexTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, validation_mode=False):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.validation_mode = validation_mode
        self.current_step = 0
        self.position = None
        self.trades_history = []
        self.entry_time = None
        self.entry_price = None
        
        # Trading metrics
        self.total_pnl = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_value = 0
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.feature_columns = self.df.select_dtypes(include=[np.number]).columns
        self.state_dim = len(self.feature_columns) + 3  # +3 for position flag, pnl, drawdown
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Load reward scaling from config
        self.reward_scaling = RL_CONFIG['reward_scaling']
        
    def _calculate_reward(self, price_change: float, done: bool) -> float:
        """Calculate scaled reward with additional factors"""
        base_reward = price_change * 100  # Convert to percentage
        
        # Scale reward based on configuration
        scaled_reward = base_reward * self.reward_scaling
        
        # Add drawdown penalty
        if self.current_drawdown < self.max_drawdown:
            self.max_drawdown = self.current_drawdown
            scaled_reward *= 0.8  # Penalty for new drawdown
        
        # Add trading frequency penalty in validation
        if self.validation_mode and len(self.trades_history) > 0:
            last_trade = self.trades_history[-1]
            steps_since_last_trade = self.current_step - last_trade['step']
            if steps_since_last_trade < 5:  # Minimum steps between trades
                scaled_reward *= 0.7  # Penalty for overtrading
        
        return scaled_reward
        
    def _get_state(self) -> np.array:
        """Create enhanced state vector"""
        current_data = self.df.iloc[self.current_step]
        
        # Get technical features
        technical_features = current_data[self.feature_columns].values
        
        # Add position flag and performance metrics
        position_flag = 0 if self.position is None else (1 if self.position == 'long' else -1)
        
        # Normalize PnL and drawdown for better scaling
        normalized_pnl = np.tanh(self.total_pnl / 100)  # Scale large PnL values
        normalized_drawdown = self.current_drawdown / (abs(self.max_drawdown) if self.max_drawdown != 0 else 1)
        
        state = np.concatenate((
            [position_flag, normalized_pnl, normalized_drawdown],
            technical_features
        ))
        return state.astype(np.float32)
    
    def _is_trading_allowed(self) -> bool:
        """Check if trading is allowed in current session"""
        current_data = self.df.iloc[self.current_step]
        return current_data['Trading_Session']
        
    def step(self, action: int) -> Tuple[np.array, float, bool, Dict]:
        # Get current price data
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        
        # Execute trading action only during trading sessions
        if self._is_trading_allowed():
            if action == 0 and self.position is None:  # Buy
                self.position = 'long'
                self.entry_price = current_price
                self.entry_time = self.df.iloc[self.current_step].name
            elif action == 1 and self.position is None:  # Sell
                self.position = 'short'
                self.entry_price = current_price
                self.entry_time = self.df.iloc[self.current_step].name
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate position outcome
        if self.position is not None:
            next_price = self.df.iloc[self.current_step]['Close']
            price_change = (next_price - self.entry_price) / self.entry_price
            
            # Calculate reward
            if self.position == 'long':
                reward = self._calculate_reward(price_change, done)
            else:  # short position
                reward = self._calculate_reward(-price_change, done)
            
            # Update metrics
            self.total_pnl += reward
            self.current_drawdown = min(0, self.total_pnl - self.peak_value)
            self.peak_value = max(self.peak_value, self.total_pnl)
            
            # Record trade
            self.trades_history.append({
                'step': self.current_step,
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': next_price,
                'reward': reward,
                'pnl': self.total_pnl,
                'drawdown': self.current_drawdown,
                'entry_time': self.entry_time,
                'exit_time': self.df.iloc[self.current_step].name,
                'London_Session': self.df.iloc[self.current_step]['London_Session'],
                'NewYork_Session': self.df.iloc[self.current_step]['NewYork_Session'],
                'Overlap_Session': self.df.iloc[self.current_step]['Overlap_Session']
            })
            
            # Close position
            self.position = None
            self.entry_price = None
            self.entry_time = None
        
        info = {
            'trades': len(self.trades_history),
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'win_rate': sum(1 for t in self.trades_history if t['reward'] > 0) / max(1, len(self.trades_history))
        }
        
        return self._get_state(), reward, done, info
        
    def reset(self):
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.trades_history = []
        self.total_pnl = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_value = 0
        return self._get_state()
