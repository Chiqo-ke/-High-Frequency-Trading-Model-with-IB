import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_session.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSession:
    def __init__(self, config=None):
        self.config = config or {
            'initial_balance': 10.0,
            'position_size_percent': 2.0,
            'pip_value': 0.1, 
            'stop_loss_pips': 5,
            'sessions': {
                'London': {'start': 8, 'end': 16},
                'NewYork': {'start': 13, 'end': 21},
                'Overlap': {'start': 13, 'end': 16}
            }
        }
        self.reset()

    def reset(self):
        """Reset trading session state"""
        self.balance = self.config['initial_balance']
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.stop_loss = None
        self.position_size = None
        self.trades = []
        self.portfolio_history = []

    def is_trading_session(self, timestamp):
        """Check if timestamp is within valid trading sessions"""
        hour = timestamp.hour
        return 8 <= hour < 21 and hour not in [16, 17]  # Exclude less liquid hours

    def get_session_label(self, timestamp):
        """Return the trading session label for a given timestamp"""
        if not self.is_trading_session(timestamp):
            return 'Off Hours'
        
        hour = timestamp.hour
        if 8 <= hour < 13:
            return 'London'
        elif 13 <= hour < 16:
            return 'Overlap'
        elif 18 <= hour < 21:
            return 'New York'
        return 'Off Hours'

    def execute_trade(self, signal_row, timestamp):
        """Execute a trade based on the signal and current market conditions"""
        current_price = signal_row['close']
        
        # Handle existing position
        if self.position:
            exit_price = None
            exit_reason = None
            
            # Check stop loss
            if ((self.position == 'long' and current_price <= self.stop_loss) or 
                (self.position == 'short' and current_price >= self.stop_loss)):
                exit_price = self.stop_loss
                exit_reason = 'stop_loss'
            
            # Check exit signals
            elif ((self.position == 'long' and signal_row['macd_cross_below']) or
                  (self.position == 'short' and signal_row['macd_cross_above'])):
                exit_price = current_price
                exit_reason = 'signal'

            # Execute exit if conditions met
            if exit_price is not None:
                self.close_position(exit_price, timestamp, exit_reason)
                return

        # Enter new position if conditions are met
        if not self.position and self.is_trading_session(timestamp):
            if signal_row['signal'] == 'buy':
                self.open_position('long', current_price, timestamp, signal_row['stop_loss_long'])
            elif signal_row['signal'] == 'sell':
                self.open_position('short', current_price, timestamp, signal_row['stop_loss_short'])

    def open_position(self, direction, price, timestamp, stop_loss):
        """Open a new trading position"""
        self.position = direction
        self.entry_price = price
        self.entry_time = timestamp
        self.stop_loss = stop_loss
        
        # Calculate position size based on risk
        risk_amount = self.balance * (self.config['position_size_percent'] / 100)
        pip_risk = abs(price - stop_loss) / self.config['pip_value']
        self.position_size = risk_amount / pip_risk
        
        logger.info(f"Opened {direction} position at {price} with stop loss at {stop_loss}")

    def close_position(self, exit_price, timestamp, reason):
        """Close an existing trading position"""
        trade_pnl = ((exit_price - self.entry_price) if self.position == 'long' 
                     else (self.entry_price - exit_price)) * self.position_size
        
        self.balance += trade_pnl
        
        self.trades.append({
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'position': self.position,
            'exit_time': timestamp,
            'exit_price': exit_price,
            'stop_loss': self.stop_loss,
            'pnl': trade_pnl,
            'exit_reason': reason,
            'session': self.get_session_label(self.entry_time)
        })
        
        logger.info(f"Closed {self.position} position at {exit_price}. PnL: {trade_pnl:.2f}")
        
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.position_size = None

    def run_simulation(self, signals_df):
        """Run trading simulation on historical data"""
        self.reset()
        start_time = signals_df.index[0]
        
        # Add 5-day warmup period
        warmup_end = start_time + timedelta(days=5)
        logger.info(f"Warmup period: {start_time} to {warmup_end}")
        
        for timestamp, row in signals_df.iterrows():
            # Skip warmup period
            if timestamp <= warmup_end:
                continue
                
            # Execute trading logic
            self.execute_trade(row, timestamp)
            
            # Track portfolio value
            equity = self.balance
            if self.position:
                unrealized_pnl = ((row['close'] - self.entry_price) if self.position == 'long' 
                                else (self.entry_price - row['close'])) * self.position_size
                equity += unrealized_pnl
            
            self.portfolio_history.append({
                'timestamp': timestamp,
                'balance': self.balance,
                'equity': equity
            })
        
        # Close any open position at the end
        if self.position:
            self.close_position(signals_df.iloc[-1]['close'], signals_df.index[-1], 'end_of_simulation')
        
        return self.create_results()

    def create_results(self):
        """Create DataFrames with trading results"""
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.set_index('entry_time', inplace=True)
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index('timestamp', inplace=True)
        
        return trades_df, portfolio_df

def generate_signals(df):
    """Generate trading signals from raw data"""
    logger.info("Generating trading signals...")
    
    # Reset existing signals
    df['signal'] = 'no_signal'
    
    # Add session information
    df['session'] = df.index.map(lambda x: TradingSession().get_session_label(x))
    df['is_trading_hours'] = df.index.map(lambda x: TradingSession().is_trading_session(x))
    
    # Generate signals during trading hours
    mask = df['is_trading_hours']
    
    # Long entry signals
    df.loc[mask & 
           (df['trend_bias'] == 'bullish') & 
           (df['price_crossed_above_ema']) & 
           (df['rsi_9'] > 30), 'signal'] = 'buy'
    
    # Short entry signals
    df.loc[mask & 
           (df['trend_bias'] == 'bearish') & 
           (df['price_crossed_below_ema']) & 
           (df['rsi_9'] < 70), 'signal'] = 'sell'
    
    # Exit signals (during all hours for risk management)
    df.loc[df['macd_cross_below'], 'signal'] = 'exit_buy'
    df.loc[df['macd_cross_above'], 'signal'] = 'exit_sell'
    
    # Calculate stop losses
    pip_value = 0.001
    stop_loss_pips = 10
    df['stop_loss_long'] = df['close'] - (stop_loss_pips * pip_value)
    df['stop_loss_short'] = df['close'] + (stop_loss_pips * pip_value)
    
    logger.info("Signal generation complete")
    return df

def main():
    """Main execution function"""
    # Load raw data
    signals_df = pd.read_csv('trading_signals.csv', parse_dates=['datetime'])
    signals_df.set_index('datetime', inplace=True)
    
    # Process signals
    logger.info("Processing trading signals...")
    signals_df = generate_signals(signals_df)
    
    # Initialize and run trading session
    session = TradingSession()
    trades_df, portfolio_df = session.run_simulation(signals_df)
    
    # Save results
    if not trades_df.empty:
        trades_df.to_csv('simulation_trades.csv')
        logger.info(f"Saved {len(trades_df)} trades to simulation_trades.csv")
        
    if not portfolio_df.empty:
        portfolio_df.to_csv('simulation_portfolio.csv')
        logger.info("Saved portfolio history to simulation_portfolio.csv")
    
    logger.info(f"Simulation completed. Final balance: {session.balance:.2f}")

if __name__ == "__main__":
    main()
