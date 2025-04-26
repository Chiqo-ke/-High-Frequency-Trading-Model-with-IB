import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import torch
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from data_processor import DataProcessor
from trading_env import ForexTradingEnv
from agent import TradingAgent
from config import DATA_DIR, MODELS_DIR, RL_CONFIG
from utils.metrics import TradingMetrics

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
    """Train the model with early stopping and enhanced metrics"""
    train_env, val_env, agent, df = setup_training(timeframe)
    
    # Initialize tracking variables
    best_reward = float('-inf')
    patience = 15
    no_improvement = 0
    training_history = []
    episode_rewards = []
    
    # Warmup replay memory
    logger.info("Warming up replay memory...")
    agent.warmup_memory(train_env, num_actions=RL_CONFIG['min_memory_size'])
    
    # Training loop
    for episode in range(episodes):
        state = train_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train(episode_reward=episode_reward)
            
            state = next_state
            episode_reward += reward
        
        # Calculate metrics
        metrics = TradingMetrics.calculate_metrics(train_env.trades_history, [episode_reward])
        training_history.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'metrics': metrics
        })
        
        # Log detailed metrics
        logger.info(f"""Episode {episode + 1}/{episodes}:
            Total Reward: {episode_reward:.2f}
            Win Rate: {metrics['win_rate']:.2%}
            Trades: {metrics['total_trades']}
            Session Performance:
                London: {metrics['session_metrics'].get('London', {}).get('win_rate', 0):.2%}
                New York: {metrics['session_metrics'].get('NewYork', {}).get('win_rate', 0):.2%}
                Overlap: {metrics['session_metrics'].get('Overlap', {}).get('win_rate', 0):.2%}
        """)
        
        # Save best model with full metrics
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = MODELS_DIR / f"best_model_{timeframe}.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'metrics': metrics,
                'hyperparameters': RL_CONFIG
            }, model_path)
            logger.info(f"New best model saved with reward: {best_reward:.2f}")
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Early stopping check
        if no_improvement >= patience:
            logger.info(f"Early stopping triggered after {patience} episodes without improvement")
            break
    
    # Plot training metrics
    plot_training_metrics(training_history, timeframe)
    return training_history

def plot_training_metrics(history: List[Dict], timeframe: str):
    """Visualize training metrics"""
    plt.style.use('seaborn')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract metrics
    episodes = [h['episode'] for h in history]
    rewards = [h['total_reward'] for h in history]
    win_rates = [h['metrics']['win_rate'] for h in history]
    
    # Plot rewards
    ax1.plot(episodes, rewards)
    ax1.set_title('Total Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    # Plot win rates
    ax2.plot(episodes, win_rates)
    ax2.set_title('Win Rate per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    
    # Plot session win rates
    session_win_rates = {
        'London': [h['metrics']['session_metrics'].get('London', {}).get('win_rate', 0) for h in history],
        'NewYork': [h['metrics']['session_metrics'].get('NewYork', {}).get('win_rate', 0) for h in history],
        'Overlap': [h['metrics']['session_metrics'].get('Overlap', {}).get('win_rate', 0) for h in history]
    }
    
    for session, rates in session_win_rates.items():
        ax3.plot(episodes, rates, label=session)
    ax3.set_title('Session Win Rates')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Win Rate')
    ax3.legend()
    
    # Plot drawdown
    drawdowns = [h['metrics'].get('max_drawdown', 0) for h in history]
    ax4.plot(episodes, drawdowns)
    ax4.set_title('Maximum Drawdown')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Drawdown')
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_{timeframe}.png')
    plt.close()

def validate_model(env: ForexTradingEnv, agent: TradingAgent, episodes: int = 10) -> Dict:
    """Validate model performance"""
    validation_rewards = []
    validation_trades = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if info.get('trade_executed', False):
                validation_trades.append(info['trade_info'])
        validation_rewards.append(env.total_pnl)
    
    return TradingMetrics.calculate_metrics(validation_trades, validation_rewards)

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
