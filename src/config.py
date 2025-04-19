import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Trading session times (UTC)
TRADING_SESSIONS = {
    'ASIAN': {
        'start': '00:00',
        'end': '09:00'
    },
    'EUROPEAN': {
        'start': '08:00',
        'end': '17:00'
    },
    'AMERICAN': {
        'start': '13:00',
        'end': '22:00'
    }
}

# RL Configuration
RL_CONFIG = {
    'learning_rate': 0.0003,  # Reduced learning rate
    'gamma': 0.95,           # Slightly reduced discount factor
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,     # Higher minimum exploration
    'epsilon_decay': 0.997,  # Slower epsilon decay
    'batch_size': 64,        # Larger batch size
    'memory_size': 100000,   # Increased replay memory
    'min_memory_size': 1000, # Minimum samples before training
    'target_update_freq': 5, # Target network update frequency
    'reward_scaling': 0.1,   # Scale rewards for better stability
    'gradient_clip': 1.0,    # Clip gradients
    'validation_episodes': 5  # Number of validation episodes
}

# Technical Indicators
TECHNICAL_INDICATORS = {
    'RSI': {'length': 14},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BB': {'length': 20, 'std': 2},
    'ATR': {'length': 14}
}
