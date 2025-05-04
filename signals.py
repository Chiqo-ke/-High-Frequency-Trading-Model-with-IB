import os
import pandas as pd
import time
from datetime import datetime, timedelta
import schedule
import pandas_ta as ta
import numpy as np

# Define constants
OUTPUT_DIR = './strategy_results'
SIGNALS_FILE = os.path.join(OUTPUT_DIR, 'signals.csv')
SIGNAL_LOG_FILE = os.path.join(OUTPUT_DIR, 'signal_log.csv')  # New constant for signal log
PIP_VALUE = 0.01  # 1 pip = 0.01 for gold
STOP_LOSS_PIPS = 20  # 20 pips stop loss (hardcoded as requested)
RECHECK_INTERVAL = 15  # Recheck every 15 seconds until we find expected data

# Data file paths
M5_DATA_FILE = './data/XAUUSD_OANDA_M5.csv'
H1_DATA_FILE = './data/XAUUSD_OANDA_H1.csv'

class SignalGenerator:
    def __init__(self):
        """Initialize the signal generator"""
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📁 Output directory created/verified: {OUTPUT_DIR}")
        
        # Initialize or load signals DataFrame
        self.signals_df = self._load_signals()
        
        # Initialize or load signal log DataFrame
        self.signal_log_df = self._load_signal_log()
        
        # Track active trades
        self.active_trades = {}  # Dictionary to track active trades
        
        # Track expected next data point times
        self.expected_next_m5 = None
        self.last_processed_m5 = None
        self.h1_data = None  # Cache for H1 data to avoid frequent reloading
        
        print("✅ Signal Generator initialized successfully!")
    
    def _load_signals(self):
        """Load existing signals or create new signals dataframe"""
        if os.path.exists(SIGNALS_FILE):
            print(f"📊 Loading existing signals from {SIGNALS_FILE}")
            signals_df = pd.read_csv(SIGNALS_FILE, parse_dates=['entry_time', 'exit_time'])
            return signals_df
        else:
            print(f"🆕 Creating new signals file at {SIGNALS_FILE}")
            # Create a new DataFrame with the required columns
            signals_df = pd.DataFrame(columns=[
                'entry_time', 'exit_time', 'entry_price', 'exit_price', 
                'position', 'stop_loss', 'pips', 'exit_reason', 'win_loss', 'status',
                'timeframe'  # Added timeframe column to track which timeframe generated the signal
            ])
            signals_df.to_csv(SIGNALS_FILE, index=False)
            return signals_df
    
    def _load_signal_log(self):
        """Load existing signal log or create new signal log dataframe"""
        if os.path.exists(SIGNAL_LOG_FILE):
            print(f"📊 Loading existing signal log from {SIGNAL_LOG_FILE}")
            signal_log_df = pd.read_csv(SIGNAL_LOG_FILE, parse_dates=['datetime'])
            return signal_log_df
        else:
            print(f"🆕 Creating new signal log file at {SIGNAL_LOG_FILE}")
            # Create a new DataFrame with the required columns
            signal_log_df = pd.DataFrame(columns=[
                'datetime',
                'price',
                'signal',  # 'buy', 'sell', or 'wait'
                'trend_bias',
                'rsi_9',
                'ema_5',
                'macd',
                'macd_signal'
            ])
            signal_log_df.to_csv(SIGNAL_LOG_FILE, index=False)
            return signal_log_df
    
    def _save_signals(self):
        """Save the signals DataFrame to CSV"""
        self.signals_df.to_csv(SIGNALS_FILE, index=False)
        print(f"💾 Signals saved to {SIGNALS_FILE}")
    
    def _save_signal_log(self):
        """Save the signal log DataFrame to CSV"""
        self.signal_log_df.to_csv(SIGNAL_LOG_FILE, index=False)
        print(f"💾 Signal log saved to {SIGNAL_LOG_FILE}")
    
    def _get_next_expected_datetime(self, current_time, timeframe):
        """Calculate the next expected datetime for the given timeframe"""
        if timeframe == 'M5':
            # Calculate the next 5-minute mark from the current time
            minutes_to_add = 5 - (current_time.minute % 5)
            if minutes_to_add == 0:
                minutes_to_add = 5
                
            # Set the next expected datetime
            next_datetime = current_time.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
                
        elif timeframe == 'H1':
            # Calculate the next hour mark
            next_datetime = current_time.replace(minute=0, second=0, microsecond=0)
            if current_time.minute > 0 or current_time.second > 0:
                next_datetime += timedelta(hours=1)
                
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        return next_datetime
    
    def _verify_data_timestamp(self, latest_data_time, expected_time, timeframe):
        """Verify that the latest data timestamp matches the expected time"""
        if timeframe == 'M5':
            # For M5, we're looking for an exact match to expected_time
            return latest_data_time == expected_time
        elif timeframe == 'H1':
            # For H1, we don't need to verify the timestamp as per requirements
            return True
        
        return False
    
    def _calculate_h1_indicators(self, price_data):
        """Calculate H1 indicators"""
        # EMA for trend bias
        price_data['ema_8'] = ta.ema(price_data['close'], length=8)
        price_data['trend_bias'] = np.where(price_data['close'] > price_data['ema_8'], 'bullish', 'bearish')
        return price_data
    
    def _calculate_m5_indicators(self, price_data, h1_trend_bias):
        """Calculate M5 indicators with H1 trend bias"""
        # Add H1 trend bias to M5 data
        price_data['trend_bias'] = h1_trend_bias
        
        # EMA
        price_data['ema_5'] = ta.ema(price_data['close'], length=5)
        
        # EMA crossover signals
        price_data['price_crossed_above_ema'] = np.where(
            (price_data['close'] > price_data['ema_5']) & 
            (price_data['close'].shift(1) <= price_data['ema_5'].shift(1)), 
            True, False
        )
        price_data['price_crossed_below_ema'] = np.where(
            (price_data['close'] < price_data['ema_5']) & 
            (price_data['close'].shift(1) >= price_data['ema_5'].shift(1)), 
            True, False
        )
        
        # RSI
        price_data['rsi_9'] = ta.rsi(price_data['close'], length=9)
        
        # MACD
        macd = ta.macd(price_data['close'], fast=8, slow=17, signal=9)
        price_data = price_data.join(macd)
        
        # MACD crossovers
        price_data['macd_cross_below'] = np.where(
            (price_data['MACD_8_17_9'] < price_data['MACDs_8_17_9']) & 
            (price_data['MACD_8_17_9'].shift(1) >= price_data['MACDs_8_17_9'].shift(1)), 
            True, False
        )
        price_data['macd_cross_above'] = np.where(
            (price_data['MACD_8_17_9'] > price_data['MACDs_8_17_9']) & 
            (price_data['MACD_8_17_9'].shift(1) <= price_data['MACDs_8_17_9'].shift(1)), 
            True, False
        )
        
        return price_data

    def _load_h1_data(self):
        """Load and process the H1 price data"""
        try:
            print(f"📊 Loading H1 data from {H1_DATA_FILE}")
            
            # Load the H1 data
            price_data = pd.read_csv(H1_DATA_FILE, parse_dates=['datetime'])
            price_data.set_index('datetime', inplace=True)
            price_data = price_data.tail(1000)  # Get last 1000 candles for indicator calculation
            
            # Calculate H1 indicators
            price_data = self._calculate_h1_indicators(price_data)
            
            # Store the H1 data for future reference
            self.h1_data = price_data
            
            # Get the latest trend bias
            latest_trend_bias = price_data['trend_bias'].iloc[-1]
            latest_time = price_data.index[-1]
            
            print(f"  H1 data timestamp: {latest_time}")
            print(f"  Current H1 trend bias: {latest_trend_bias}")
            
            return latest_trend_bias
            
        except Exception as e:
            print(f"❌ Error loading H1 price data: {e}")
            return None

    def _load_m5_data(self, expected_time=None):
        """Load and process the M5 price data, optionally checking for an expected timestamp"""
        try:
            print(f"📊 Loading M5 data from {M5_DATA_FILE}")
            
            # Load the M5 data
            price_data = pd.read_csv(M5_DATA_FILE, parse_dates=['datetime'])
            price_data.set_index('datetime', inplace=True)
            
            # Get the latest timestamp
            latest_time = price_data.index[-1]
            print(f"  Latest M5 data timestamp: {latest_time}")
            
            # Always set the next expected time based on the latest data point
            self.expected_next_m5 = latest_time + timedelta(minutes=5)
            print(f"⏰ Next expected M5 check at: {self.expected_next_m5}")
            
            # If we're looking for a specific timestamp and it's not present
            if expected_time is not None:
                if not any(price_data.index == expected_time):
                    print(f"⏳ Waiting for M5 data at {expected_time}, latest available is {latest_time}")
                    return None, None
                data_point = price_data.loc[expected_time]
                relevant_time = expected_time
            else:
                data_point = price_data.iloc[-1]
                relevant_time = latest_time
            
            # Get the latest H1 trend bias
            h1_trend_bias = self._load_h1_data()
            if h1_trend_bias is None:
                return None, None
            
            # Calculate M5 indicators for the last 100 candles
            m5_subset = price_data.tail(100)
            m5_subset = self._calculate_m5_indicators(m5_subset, h1_trend_bias)
            
            # Get the specific row we need (latest or expected)
            if expected_time is not None:
                latest_data = m5_subset.loc[expected_time]
            else:
                latest_data = m5_subset.iloc[-1]
            
            # Update last processed time
            self.last_processed_m5 = relevant_time
            
            return latest_data, relevant_time
            
        except Exception as e:
            print(f"❌ Error loading M5 price data: {e}")
            return None, None

    def _log_signal_analysis(self, latest_data, latest_time, signal):
        """Log the signal analysis for the current timeframe"""
        # Create new log entry
        new_log = {
            'datetime': latest_time,
            'price': latest_data['close'],
            'signal': signal if signal else 'wait',  # If no signal, mark as 'wait'
            'trend_bias': latest_data['trend_bias'],
            'rsi_9': latest_data['rsi_9'],
            'ema_5': latest_data['ema_5'],
            'macd': latest_data['MACD_8_17_9'],
            'macd_signal': latest_data['MACDs_8_17_9']
        }

        # Add to signal log DataFrame
        self.signal_log_df = pd.concat([self.signal_log_df, pd.DataFrame([new_log])], ignore_index=True)
        
        # Save updated signal log to CSV
        self._save_signal_log()

    def _generate_signals(self, latest_data, latest_time):
        """Generate trading signals based on the calculated indicators"""
        signal = None
        exit_buy = False
        exit_sell = False

        try:
            # Entry signals
            if (latest_data['trend_bias'] == 'bullish' and
                latest_data['price_crossed_above_ema'] and
                latest_data['rsi_9'] > 30):
                signal = 'buy'
            elif (latest_data['trend_bias'] == 'bearish' and
                 latest_data['price_crossed_below_ema'] and
                 latest_data['rsi_9'] < 70):
                signal = 'sell'

            # Log the signal analysis
            self._log_signal_analysis(latest_data, latest_time, signal)

            # Exit signals
            exit_buy = latest_data['macd_cross_below']
            exit_sell = latest_data['macd_cross_above']

            print(f"Signal generated: {signal}, Exit Buy: {exit_buy}, Exit Sell: {exit_sell}")
            return signal, exit_buy, exit_sell

        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return None, False, False

    def _process_entries(self, signal, latest_data, latest_time):
        """Process any new entry signals"""
        if signal in ['buy', 'sell']:
            current_price = latest_data['close']
            
            # Determine stop loss based on position
            stop_loss = current_price - (STOP_LOSS_PIPS * PIP_VALUE) if signal == 'buy' else current_price + (STOP_LOSS_PIPS * PIP_VALUE)
            
            # Create new entry record
            new_entry = {
                'entry_time': latest_time,
                'exit_time': pd.NaT,
                'entry_price': current_price,
                'exit_price': None,
                'position': 'long' if signal == 'buy' else 'short',
                'stop_loss': stop_loss,
                'pips': None,
                'exit_reason': None,
                'win_loss': None,
                'status': 'active',
                'timeframe': 'M5'  # All entries are from M5 timeframe
            }
            
            # Add to signals DataFrame
            self.signals_df = pd.concat([self.signals_df, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Add to active trades dictionary for easy access
            trade_id = f"M5_{str(latest_time)}"
            self.active_trades[trade_id] = len(self.signals_df) - 1  # Index in the DataFrame
            
            # Set the expected next data point (5 minutes after this entry)
            self.expected_next_m5 = latest_time + timedelta(minutes=5)
            
            print(f"🔔 New {signal.upper()} entry at {latest_time}: Price={current_price}, Stop Loss={stop_loss}")
            print(f"⏰ Next expected M5 check at: {self.expected_next_m5}")
            
            # Save updated signals to CSV
            self._save_signals()
    
    def _process_exits(self, exit_buy, exit_sell, latest_data, latest_time):
        """Process any exit signals for active trades"""
        current_price = latest_data['close']
        
        # Create a list of trades to remove from active_trades
        to_remove = []
        
        # Check each active trade for exit conditions
        for trade_id, index in self.active_trades.items():
            # Get the trade data
            trade = self.signals_df.iloc[index]
            
            # Skip if trade is already completed
            if trade['status'] != 'active':
                to_remove.append(trade_id)
                continue
            
            # Check for signal-based exit
            exit_price = None
            exit_reason = None
            
            if (trade['position'] == 'long' and exit_buy) or (trade['position'] == 'short' and exit_sell):
                exit_price = current_price
                exit_reason = 'signal'
                
            # Check for stop loss hit
            elif (trade['position'] == 'long' and current_price <= trade['stop_loss']) or \
                 (trade['position'] == 'short' and current_price >= trade['stop_loss']):
                exit_price = trade['stop_loss']
                exit_reason = 'stop_loss'
            
            # If we have an exit condition
            if exit_price is not None:
                # Calculate pips gained/lost
                pips = ((exit_price - trade['entry_price']) / PIP_VALUE if trade['position'] == 'long' 
                       else (trade['entry_price'] - exit_price) / PIP_VALUE)
                
                # Update the trade record
                self.signals_df.at[index, 'exit_price'] = exit_price
                self.signals_df.at[index, 'exit_time'] = latest_time
                self.signals_df.at[index, 'exit_reason'] = exit_reason
                self.signals_df.at[index, 'pips'] = pips
                self.signals_df.at[index, 'win_loss'] = 'win' if pips > 0 else 'loss'
                self.signals_df.at[index, 'status'] = 'completed'
                
                print(f"🔔 Trade CLOSED at {latest_time}: Entry={trade['entry_price']}, Exit={exit_price}, Pips={pips:.2f}, Result={'WIN' if pips > 0 else 'LOSS'}")
                
                # Set the expected next data point (5 minutes after this exit)
                self.expected_next_m5 = latest_time + timedelta(minutes=5)
                print(f"⏰ Next expected M5 check at: {self.expected_next_m5}")
                
                # Mark for removal from active trades
                to_remove.append(trade_id)
        
        # Remove completed trades from active_trades dictionary
        for trade_id in to_remove:
            self.active_trades.pop(trade_id, None)
        
        # If any trades were updated, save the signals file
        if to_remove:
            self._save_signals()
    
    def _print_summary(self):
        """Print a summary of current trading performance"""
        if len(self.signals_df) == 0:
            print("📊 No trades yet")
            return
        
        completed_trades = self.signals_df[self.signals_df['status'] == 'completed']
        active_trades = self.signals_df[self.signals_df['status'] == 'active']
        
        print("\n📈 Trading Performance Summary:")
        print(f"  Total Completed Trades: {len(completed_trades)}")
        
        if len(completed_trades) > 0:
            total_pips = completed_trades['pips'].sum()
            win_rate = (len(completed_trades[completed_trades['pips'] > 0]) / len(completed_trades) * 100)
            
            print(f"  Total Pips: {total_pips:.2f}")
            print(f"  Win Rate: {win_rate:.2f}%")
        
        print(f"  Active Trades: {len(active_trades)}")
        
        # Add timeframe-specific stats
        if 'timeframe' in self.signals_df.columns:
            m5_trades = completed_trades[completed_trades['timeframe'] == 'M5']
            
            if len(m5_trades) > 0:
                m5_pips = m5_trades['pips'].sum()
                m5_win_rate = (len(m5_trades[m5_trades['pips'] > 0]) / len(m5_trades) * 100)
                print(f"  M5 Trades: {len(m5_trades)}, Pips: {m5_pips:.2f}, Win Rate: {m5_win_rate:.2f}%")
        
        print("")
    
    def run_check(self):
        """Run the signal check based on current state"""
        now = datetime.now()
        print(f"\n⏰ Running signal check at {now}")
        
        # If we have an expected next data point, check for it
        if self.expected_next_m5 is not None:
            print(f"  Looking for expected M5 data at {self.expected_next_m5}")
            
            # Try to load the data at the expected timestamp
            latest_data, latest_time = self._load_m5_data(expected_time=self.expected_next_m5)
            
            if latest_data is None:
                # If the expected data isn't available yet, schedule a recheck
                print(f"⏳ Expected M5 data not found yet. Will check again in {RECHECK_INTERVAL} seconds.")
                
                # Keep the expected time the same for the next check
                return False
            
            # If we've successfully loaded the expected data
            print(f"✅ Found expected M5 data at {latest_time}")
            
            # Now process this data point
            signal, exit_buy, exit_sell = self._generate_signals(latest_data, latest_time)
            
            # Process exits first
            self._process_exits(exit_buy, exit_sell, latest_data, latest_time)
            
            # Process entries
            self._process_entries(signal, latest_data, latest_time)
            
            # Print summary
            self._print_summary()
            
            # Data was found and processed, clear the expected time if no new one was set
            if self.expected_next_m5 == latest_time + timedelta(minutes=5):
                # If we didn't set a new expected time during processing, clear it
                self.expected_next_m5 = None
                
            return True
            
        else:
            # If we don't have an expected next data point, just check the latest data
            print("  Checking latest available M5 data")
            
            # Load the latest data
            latest_data, latest_time = self._load_m5_data()
            
            if latest_data is None:
                # If no data is available at all
                print("❌ No M5 data available. Will check again later.")
                return False
            
            # If we've already processed this data point, don't process it again
            if self.last_processed_m5 is not None and self.last_processed_m5 >= latest_time:
                print(f"ℹ️ Already processed data at {latest_time}. Waiting for new data.")
                
                # Set the expected next data point to 5 minutes after the last processed
                self.expected_next_m5 = self.last_processed_m5 + timedelta(minutes=5)
                print(f"⏰ Next expected M5 check at: {self.expected_next_m5}")
                
                return False
            
            # Process the latest data
            print(f"✅ Processing latest M5 data at {latest_time}")
            
            signal, exit_buy, exit_sell = self._generate_signals(latest_data, latest_time)
            
            # Process exits first
            self._process_exits(exit_buy, exit_sell, latest_data, latest_time)
            
            # Process entries
            self._process_entries(signal, latest_data, latest_time)
            
            # Print summary
            self._print_summary()
            
            # If we haven't set a new expected time during processing, set it now
            if self.expected_next_m5 is None:
                self.expected_next_m5 = latest_time + timedelta(minutes=5)
                print(f"⏰ Next expected M5 check at: {self.expected_next_m5}")
            
            return True
    
    def run_scheduled_check(self):
        """Run a scheduled check that potentially reschedules itself"""
        success = self.run_check()
        
        if not success:
            # If the check wasn't successful (e.g., expected data not found),
            # schedule another check in RECHECK_INTERVAL seconds
            print(f"⏰ Scheduling recheck in {RECHECK_INTERVAL} seconds")
            schedule.clear()
            schedule.every(RECHECK_INTERVAL).seconds.do(self.run_scheduled_check)
        else:
            # If the check was successful, schedule the next check based on expected_next_m5
            if self.expected_next_m5 is not None:
                # Clear any existing schedule
                schedule.clear()
                
                # Calculate time until next expected data point
                now = datetime.now()
                time_to_next = (self.expected_next_m5 - now).total_seconds()
                
                # If next data point is in the past, check again in RECHECK_INTERVAL seconds
                if time_to_next <= 0:
                    print(f"⏰ Next expected time is in the past. Rechecking in {RECHECK_INTERVAL} seconds")
                    schedule.every(RECHECK_INTERVAL).seconds.do(self.run_scheduled_check)
                else:
                    # Otherwise, schedule for the expected time
                    print(f"⏰ Scheduling next check in {time_to_next:.2f} seconds")
                    schedule.every(time_to_next).seconds.do(self.run_scheduled_check)
            else:
                # If no expected next time, check every 5 minutes
                print(f"⏰ No specific next time expected. Checking again in 5 minutes")
                schedule.clear()
                schedule.every(5).minutes.do(self.run_scheduled_check)

def main():
    """Main function to set up and run the automated trading system"""
    print("🤖 Starting Automated Trading Signal Generator")
    print(f"⏱️ Running with automatic M5 scheduling and continuous H1 monitoring")
    
    # Create an instance of SignalGenerator
    generator = SignalGenerator()
    
    # Run the first check immediately
    generator.run_scheduled_check()
    
    # Print the next scheduled run
    next_run = schedule.next_run()
    if next_run:
        print(f"📅 Next scheduled run: {next_run}")
    
    # Keep the script running continuously
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()