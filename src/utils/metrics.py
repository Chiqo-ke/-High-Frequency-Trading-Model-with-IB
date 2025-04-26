import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict

class TradingMetrics:
    @staticmethod
    def calculate_metrics(trades_history: List[Dict], episode_rewards: List[float]) -> Dict:
        """Calculate comprehensive trading metrics"""
        if not trades_history:
            return {}

        # Session-specific metrics
        session_trades = defaultdict(list)
        for trade in trades_history:
            if trade.get('London_Session'):
                session_trades['London'].append(trade)
            if trade.get('NewYork_Session'):
                session_trades['NewYork'].append(trade)
            if trade.get('Overlap_Session'):
                session_trades['Overlap'].append(trade)

        metrics = {
            # Overall performance
            'total_reward': sum(episode_rewards),
            'mean_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'max_drawdown': TradingMetrics._calculate_max_drawdown(episode_rewards),
            
            # Trading metrics
            'total_trades': len(trades_history),
            'win_rate': sum(1 for t in trades_history if t['reward'] > 0) / len(trades_history),
            'profit_factor': TradingMetrics._calculate_profit_factor(trades_history),
            'avg_trade_duration': np.mean([t['duration'] for t in trades_history if 'duration' in t]),
            
            # Session performance
            'session_metrics': {
                session: {
                    'trades': len(trades),
                    'win_rate': sum(1 for t in trades if t['reward'] > 0) / max(len(trades), 1),
                    'total_reward': sum(t['reward'] for t in trades),
                    'avg_reward': np.mean([t['reward'] for t in trades]) if trades else 0
                }
                for session, trades in session_trades.items()
            },
            
            # Risk metrics
            'sharpe_ratio': TradingMetrics._calculate_sharpe_ratio(episode_rewards),
            'max_consecutive_losses': TradingMetrics._calculate_max_consecutive(trades_history, False),
            'max_consecutive_wins': TradingMetrics._calculate_max_consecutive(trades_history, True)
        }
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown(rewards: List[float]) -> float:
        cumulative = np.cumsum(rewards)
        max_dd = 0
        peak = cumulative[0]
        
        for value in cumulative[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    @staticmethod
    def _calculate_profit_factor(trades: List[Dict]) -> float:
        winning_trades = sum(t['reward'] for t in trades if t['reward'] > 0)
        losing_trades = abs(sum(t['reward'] for t in trades if t['reward'] < 0))
        return winning_trades / losing_trades if losing_trades != 0 else float('inf')
    
    @staticmethod
    def _calculate_sharpe_ratio(rewards: List[float], risk_free_rate=0.0) -> float:
        if not rewards:
            return 0.0
        returns = np.array(rewards)
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def _calculate_max_consecutive(trades: List[Dict], winning: bool) -> int:
        max_streak = current_streak = 0
        for trade in trades:
            is_win = trade['reward'] > 0
            if is_win == winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak