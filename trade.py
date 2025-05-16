import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pandas_ta as ta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_sim.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSimulator:
    def __init__(self, initial_balance=100, lot_size=0.1):
        self.balance = initial_balance
        self.lot_size = lot_size  # 0.1 lots = 10,000 units
        self.pip_value = (10 * self.lot_size)  # For EURUSD, 1 standard lot = $10 per pip
        self.stop_loss_pips = 25
        self.current_trade = None
        self.trades_history = []
    
    def calculate_position_size(self):
        """Calculate position size based on risk management"""
        risk_per_trade = self.balance * 0.02  # 2% risk per trade
        risk_per_pip = risk_per_trade / self.stop_loss_pips
        return min(self.lot_size, round(risk_per_pip / 10, 2))  # Divide by $10 (1 standard lot pip value)
    
    def open_trade(self, entry_price, direction, timestamp):
        """Open a new trade position"""
        if self.current_trade is not None:
            return False
            
        stop_loss = entry_price - (self.stop_loss_pips * 0.0001) if direction == 'buy' else \
                    entry_price + (self.stop_loss_pips * 0.0001)
        
        position_size = self.calculate_position_size()
        
        self.current_trade = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'status': 'open'
        }
        
        logger.info(f"Opened {direction} position: Entry={entry_price}, SL={stop_loss}, Size={position_size}")
        return True
    
    def close_trade(self, exit_price, timestamp, reason="signal"):
        """Close current trade and calculate profit/loss"""
        if self.current_trade is None:
            return
            
        # Calculate pip difference (0.0001 = 1 pip)
        pip_difference = (exit_price - self.current_trade['entry_price']) * 10000
        if self.current_trade['direction'] == 'sell':
            pip_difference = -pip_difference
            
        # Calculate profit: pips × pip_value (adjusted for position size)
        profit = pip_difference * self.current_trade['position_size'] * 10  # $10 per pip per standard lot
        self.balance += profit
        
        trade_result = {
            **self.current_trade,
            'exit_time': timestamp,
            'exit_price': exit_price,
            'profit': profit,
            'pips': pip_difference,
            'close_reason': reason
        }
        
        self.trades_history.append(trade_result)
        logger.info(f"Closed trade: Pips={pip_difference:.1f}, Profit=${profit:.2f}, Balance=${self.balance:.2f}")
        
        self.current_trade = None
        return trade_result
    
    def check_stop_loss(self, current_price, timestamp):
        """Check if stop loss has been hit"""
        if self.current_trade is None:
            return False
            
        if self.current_trade['direction'] == 'buy':
            if current_price <= self.current_trade['stop_loss']:
                self.close_trade(current_price, timestamp, "stop_loss")
                return True
        else:  # sell
            if current_price >= self.current_trade['stop_loss']:
                self.close_trade(current_price, timestamp, "stop_loss")
                return True
        return False
    
    def process_signals(self, m5_data, h1_data):
        """Process trading signals and execute trades"""
        results = []
        
        for idx in m5_data.index:
            current_price = m5_data.loc[idx, 'close']
            current_time = idx
            
            # Check stop loss first
            if self.check_stop_loss(current_price, current_time):
                continue
            
            # Get the latest H1 trend
            h1_row = h1_data[h1_data.index <= current_time].iloc[-1]
            trend_bias = h1_row['trend_bias']
            
            # Current M5 row
            m5_row = m5_data.loc[idx]
            
            # Entry conditions with RSI confirmation
            if self.current_trade is None:  # No open position
                if trend_bias == 'bullish' and m5_row['price_crossed_above_ema'] and m5_row['rsi_9'] > 30:
                    self.open_trade(current_price, 'buy', current_time)
                elif trend_bias == 'bearish' and m5_row['price_crossed_below_ema'] and m5_row['rsi_9'] < 70:
                    self.open_trade(current_price, 'sell', current_time)
            
            # Exit conditions based on MACD crossovers
            elif self.current_trade['direction'] == 'buy':
                if m5_row['macd_cross_below']:
                    self.close_trade(current_price, current_time, "signal")
            elif self.current_trade['direction'] == 'sell':
                if m5_row['macd_cross_above']:
                    self.close_trade(current_price, current_time, "signal")
        
        return self.trades_history

def prepare_data(m5_data, h1_data):
    """Prepare data with necessary indicators"""
    # H1 data: Trend bias with 8-period EMA
    h1_data['ema8'] = ta.ema(h1_data['close'], length=8)
    h1_data['trend_bias'] = np.where(h1_data['close'] > h1_data['ema8'], 'bullish', 'bearish')
    
    # M5 data: Indicators for entry and exit
    m5_data['ema5'] = ta.ema(m5_data['close'], length=5)
    m5_data['price_crossed_above_ema'] = (m5_data['close'] > m5_data['ema5']) & (m5_data['close'].shift(1) < m5_data['ema5'].shift(1))
    m5_data['price_crossed_below_ema'] = (m5_data['close'] < m5_data['ema5']) & (m5_data['close'].shift(1) > m5_data['ema5'].shift(1))
    m5_data['rsi_9'] = ta.rsi(m5_data['close'], length=9)
    macd = ta.macd(m5_data['close'], fast=8, slow=17, signal=9)
    m5_data['macd_line'] = macd['MACD_8_17_9']
    m5_data['macd_signal'] = macd['MACDs_8_17_9']
    m5_data['macd_cross_below'] = (m5_data['macd_line'] < m5_data['macd_signal']) & (m5_data['macd_line'].shift(1) > m5_data['macd_signal'].shift(1))
    m5_data['macd_cross_above'] = (m5_data['macd_line'] > m5_data['macd_signal']) & (m5_data['macd_line'].shift(1) < m5_data['macd_signal'].shift(1))

def run_simulation():
    """Run trading simulation on historical data"""
    try:
        # Load data
        m5_data = pd.read_csv('M5.csv', index_col=0, parse_dates=True)
        h1_data = pd.read_csv('H1.csv', index_col=0, parse_dates=True)
        
        # Prepare data with indicators
        prepare_data(m5_data, h1_data)
        
        # Initialize simulator
        simulator = TradingSimulator(initial_balance=100, lot_size=0.1)
        
        # Run simulation
        trades = simulator.process_signals(m5_data, h1_data)
        
        # Calculate statistics
        if trades:
            df_trades = pd.DataFrame(trades)
            total_trades = len(trades)
            winning_trades = len(df_trades[df_trades['profit'] > 0])
            total_profit = df_trades['profit'].sum()
            win_rate = (winning_trades / total_trades) * 100
            
            logger.info(f"\nSimulation Results:")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Winning Trades: {winning_trades}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
            logger.info(f"Total Profit: ${total_profit:.2f}")
            logger.info(f"Final Balance: ${simulator.balance:.2f}")
            
            # Save trades to CSV
            df_trades.to_csv('trades_history.csv')
        else:
            logger.warning("No trades were executed during simulation")
            
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}", exc_info=True)

if __name__ == '__main__':
    run_simulation()