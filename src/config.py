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
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'batch_size': 32,
}

# Technical Indicators
TECHNICAL_INDICATORS = {
    'RSI': {'length': 14},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BB': {'length': 20, 'std': 2},
    'ATR': {'length': 14}
}
