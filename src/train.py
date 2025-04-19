import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import torch

from data_processor import DataProcessor
from trading_env import ForexTradingEnv
from agent import TradingAgent
from config import DATA_DIR, MODELS_DIR, RL_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_training(timeframe: str):
    """Initialize training components with validation split"""
    # Load and process data
    data_file = DATA_DIR / f"{timeframe}.csv"
    processor = DataProcessor()
    df = processor.load_data(data_file)
    df = processor.add_technical_indicators(df)
    df = df.dropna()
    
    # Split data into train/validation
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # Create environments
    train_env = ForexTradingEnv(train_df, validation_mode=False)
    val_env = ForexTradingEnv(val_df, validation_mode=True)
    
    # Initialize agent
    agent = TradingAgent(
        state_size=train_env.state_dim,
        action_size=train_env.action_space.n
    )
    
    return train_env, val_env, agent, df

def train_model(timeframe: str, episodes: int = 1000):
    """Train the trading model with improved stability measures"""
    # Setup environments
    train_env, val_env, agent, df = setup_training(timeframe)
    
    # Initialize metrics
    best_reward = float('-inf')
    patience = 20
    no_improvement = 0
    training_history = []
    
    # Warmup replay memory
    logger.info("Warming up replay memory...")
    agent.warmup_memory(train_env, num_actions=RL_CONFIG['min_memory_size'])
    
    for episode in range(episodes):
        # Training episode
        state = train_env.reset()
        total_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train(episode_reward=total_reward)
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
        
        # Validation phase every 10 episodes
        if episode % 10 == 0:
            val_rewards = []
            for _ in range(RL_CONFIG['validation_episodes']):
                val_reward = validate_episode(val_env, agent)
                val_rewards.append(val_reward)
            avg_val_reward = np.mean(val_rewards)
            
            # Save model if validation improves
            if avg_val_reward > best_reward:
                best_reward = avg_val_reward
                model_path = MODELS_DIR / f"best_model_{timeframe}.pth"
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'best_reward': best_reward,
                    'config': RL_CONFIG
                }, model_path)
                logger.info(f"New best model saved with validation reward: {best_reward:.2f}")
                no_improvement = 0
            else:
                no_improvement += 1
        
        # Log episode metrics
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        metrics = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'avg_loss': avg_loss,
            'epsilon': agent.epsilon,
            'trades': train_env.trades_history[-1]['trades'] if train_env.trades_history else 0,
            'win_rate': train_env.trades_history[-1]['win_rate'] if train_env.trades_history else 0
        }
        training_history.append(metrics)
        
        # Print progress
        logger.info(
            f"Episode {episode + 1}/{episodes}, "
            f"Total Reward: {total_reward:.2f}, "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Epsilon: {agent.epsilon:.3f}, "
            f"Win Rate: {metrics['win_rate']:.2%}"
        )
        
        # Early stopping
        if no_improvement >= patience:
            logger.info(f"Early stopping triggered after {patience} episodes without improvement")
            break
    
    return training_history

def validate_episode(env, agent):
    """Run a validation episode"""
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    
    return total_reward

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
