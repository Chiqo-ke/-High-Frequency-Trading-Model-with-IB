import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from clean import (process_new_order_data, calculate_indicators, 
                  create_signals_dataframe, store_dataframe_as_csv)
import pandas as pd
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSVHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_processed = {}
        self.processing_lock = False
        self.last_modified = {}  # Track last modification time for each file
    
    def on_modified(self, event):
        if not event.src_path.endswith('.csv') or self.processing_lock:
            return
            
        try:
            self.processing_lock = True
            file_name = os.path.basename(event.src_path)
            timeframe = file_name.replace('.csv', '')
            
            # Check if this is a valid timeframe file
            if timeframe not in ['H1', 'M5']:
                return
                
            # Prevent duplicate processing of the same modification
            current_mtime = os.path.getmtime(event.src_path)
            if timeframe in self.last_modified:
                if current_mtime <= self.last_modified[timeframe]:
                    logger.debug(f"Skipping duplicate modification for {timeframe}")
                    return
                    
            self.last_modified[timeframe] = current_mtime
            
            # Add delay to ensure file is completely written
            time.sleep(1)
            
            logger.info(f"Detected update in {timeframe} data")
            self.process_update(timeframe)
            
        except Exception as e:
            logger.error(f"Error in on_modified: {e}")
        finally:
            self.processing_lock = False
    
    def process_update(self, timeframe):
        """Process updates with proper error handling and data validation"""
        try:
            # Load the raw data first
            file_path = os.path.join('data', f'{timeframe}.csv')
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return
                
            try:
                raw_df = pd.read_csv(file_path)
                if raw_df.empty:
                    logger.warning(f"Empty data file: {file_path}")
                    return
            except Exception as e:
                logger.error(f"Error reading CSV file {file_path}: {e}")
                return
            
            # Process the new data
            df = process_new_order_data(raw_df, timeframe)
            if df is None:
                logger.warning(f"No new data to process for {timeframe}")
                return
            
            # Calculate new indicators
            try:
                h1_data, m5_data = calculate_indicators()
                if h1_data is None and m5_data is None:
                    logger.error("Failed to calculate indicators")
                    return
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                return
            
            # Generate trading signals
            try:
                signals_df = create_signals_dataframe()
                if signals_df is None:
                    logger.error("Failed to create signals dataframe")
                    return
                
                # Apply trading strategy
                signals_df = self.apply_trading_strategy(signals_df)
                if signals_df is None:
                    logger.error("Failed to apply trading strategy")
                    return
                
                # Save processed signals
                if store_dataframe_as_csv('processed_signals', signals_df):
                    logger.info(f"Successfully processed {timeframe} update and generated signals")
                else:
                    logger.error("Failed to save processed signals")
                
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
            
        except Exception as e:
            logger.error(f"Error processing {timeframe} update: {e}")
            
    def apply_trading_strategy(self, df):
        """Apply the trading strategy with error handling"""
        try:
            if df is None or df.empty:
                logger.warning("No data available for trading strategy")
                return None
            
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Validate required columns
            required_columns = ['trend_bias', 'price_crossed_above_ema', 'price_crossed_below_ema', 
                              'rsi_9', 'macd_cross_below', 'macd_cross_above']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Apply signals with error handling
            df['signal'] = 'no_signal'
            
            try:
                # Long entry conditions
                long_condition = (
                    (df['trend_bias'] == 'bullish') &
                    (df['price_crossed_above_ema']) &
                    (df['rsi_9'] > 30)
                )
                df.loc[long_condition, 'signal'] = 'buy'
                
                # Short entry conditions
                short_condition = (
                    (df['trend_bias'] == 'bearish') &
                    (df['price_crossed_below_ema']) &
                    (df['rsi_9'] < 70)
                )
                df.loc[short_condition, 'signal'] = 'sell'
                
                # Exit conditions using MACD
                df.loc[df['macd_cross_below'], 'signal'] = 'exit_buy'
                df.loc[df['macd_cross_above'], 'signal'] = 'exit_sell'
                
            except Exception as e:
                logger.error(f"Error applying trading conditions: {e}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error in apply_trading_strategy: {e}")
            return None

def start_monitoring(data_dir='data'):
    """Start monitoring CSV files for changes"""
    logger.info(f"Starting to monitor {data_dir} for CSV updates...")
    
    event_handler = CSVHandler()
    observer = Observer()
    observer.schedule(event_handler, data_dir, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped monitoring CSV files")
    
    observer.join()

if __name__ == "__main__":
    start_monitoring()
