import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import torch

from data_processor import DataProcessor
from trading_env import ForexTradingEnv
from agent import TradingAgent
from config import DATA_DIR, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_training(timeframe: str):
    """Initialize training components for a specific timeframe"""
    # Load and process data
    data_file = DATA_DIR / f"{timeframe}.csv"
    processor = DataProcessor()
    df = processor.load_data(data_file)
    
    # Add debug logging
    logger.info(f"Original dataframe shape: {df.shape}")
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # Process indicators
    df = processor.add_technical_indicators(df)
    df = df.dropna()  # Remove rows with NaN values
    
    # Log processed data info
    logger.info(f"Processed dataframe shape: {df.shape}")
    logger.info(f"Processed columns: {df.columns.tolist()}")
    
    # Calculate actual feature dimension
    num_features = len(df.columns) - 1  # Exclude 'Time' column
    
    # Create environment and agent with correct dimensions
    env = ForexTradingEnv(df, state_dim=num_features)
    agent = TradingAgent(state_size=num_features, action_size=3)
    
    return env, agent, df

def train_model(timeframe: str, episodes: int = 1000):
    """Main training loop"""
    env, agent, df = setup_training(timeframe)
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action and perform step
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
        
        # Log progress
        logger.info(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            model_path = MODELS_DIR / f"best_model_{timeframe}.pth"
            torch.save(agent.model.state_dict(), model_path)
            logger.info(f"New best model saved with reward: {best_reward:.2f}")

def main():
    """Train models for all timeframes"""
    timeframes = ['M5', 'M15', 'M30', 'H1', 'H4']
    
    for timeframe in timeframes:
        logger.info(f"Starting training for {timeframe} timeframe")
        try:
            train_model(timeframe)
        except Exception as e:
            logger.error(f"Error training {timeframe}: {str(e)}")
            continue

if __name__ == "__main__":
    # Create necessary directories
    MODELS_DIR.mkdir(exist_ok=True)
    main()
