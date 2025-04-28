# trading_view_data_fetcher.py
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging
from tvDatafeed import TvDatafeed, Interval
from .redis_store import RedisStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingViewDataFetcher:
    def __init__(self, username=None, password=None):
        """Initialize the TradingView data fetcher with optional credentials."""
        logger.info("Connecting to TradingView...")
        try:
            if username and password:
                self.tv = TvDatafeed(username=username, password=password)
                logger.info("Connection established with credentials!")
            else:
                self.tv = TvDatafeed()
                logger.info("Connection established without login (some data may be limited)")
                
            # Initialize Redis store
            self.redis_store = RedisStore()
            
            # Ensure output directory exists
            os.makedirs('data', exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to connect to TradingView: {e}")
            raise
    
    def fetch_historical_data(self, symbol, exchange, interval, days_back=7):
        """Fetch historical data for specified interval and days back."""
        logger.info(f"Fetching historical {interval} data for {symbol}...")
        
        try:
            # Calculate exact timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Map our interval names to tvDatafeed Interval objects
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute
            }
            
            # Calculate how many bars we need based on interval
            n_bars = self.calculate_n_bars(interval, start_time, end_time)
            logger.debug(f"Requesting {n_bars} bars for {interval}")
            
            # Fetch the data
            df = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval_map[interval],
                n_bars=n_bars
            )
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} on {exchange} with interval {interval}")
                return None
                
            # Reset index to make datetime a column and format for CSV
            df = df.reset_index()
            
            # Save to CSV with simplified naming
            filepath = os.path.join('data', f'{interval}.csv')
            df.to_csv(filepath, index=False)
            logger.info(f"Saved historical data to {filepath} ({len(df)} rows)")
            
            # Save to Redis
            redis_key = f"{symbol}_{exchange}_{interval}"
            self.redis_store.save_data(redis_key, df)
            logger.info(f"Saved historical data to Redis with key: {redis_key}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def calculate_n_bars(self, interval, start_time, end_time):
        """Calculate the number of bars needed based on the interval."""
        if interval == 'H1':
            return int((end_time - start_time).total_seconds() / 3600) + 10  # Add buffer
        elif interval == 'M5':
            return int((end_time - start_time).total_seconds() / 300) + 10  # Add buffer
        return 0
    
    def is_candle_complete(self, interval, datetime_val):
        """Check if the current candle is complete based on its datetime"""
        try:
            current_time = datetime.now()
            
            # Convert to datetime if string
            if isinstance(datetime_val, str):
                candle_time = pd.to_datetime(datetime_val)
            else:
                candle_time = datetime_val
                
            # Remove timezone info if present to avoid comparison issues
            if hasattr(candle_time, 'tz_localize'):
                candle_time = candle_time.tz_localize(None)
            
            # Calculate when the next candle should start
            if interval == 'H1':
                next_candle = candle_time + pd.Timedelta(hours=1)
            elif interval == 'M5':
                next_candle = candle_time + pd.Timedelta(minutes=5)
            else:
                logger.warning(f"Unknown interval: {interval}")
                return False

            # Debug information
            logger.debug(f"Candle time: {candle_time}, Next candle: {next_candle}, Current time: {current_time}")
            is_complete = current_time >= next_candle
            
            if is_complete:
                logger.debug(f"Candle at {candle_time} is complete")
            else:
                logger.debug(f"Candle at {candle_time} is still forming ({(next_candle - current_time).total_seconds():.1f} seconds remaining)")
                
            return is_complete
            
        except Exception as e:
            logger.error(f"Error checking if candle is complete: {e}")
            return False

    def update_data(self, symbol, exchange, interval):
        """Update data for the specified interval"""
        logger.info(f"Updating {interval} data for {symbol}...")

        try:
            # Map intervals to tvDatafeed Interval objects
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute
            }

            # Get the latest data
            latest_df = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval_map[interval],
                n_bars=10  # Increased to get more historical bars
            )
            
            if latest_df is None or latest_df.empty:
                logger.warning(f"No data returned when updating {interval} data")
                return None

            latest_df = latest_df.reset_index()
            
            # Log the first and last data timestamps
            if not latest_df.empty:
                first_time = latest_df['datetime'].iloc[0]
                last_time = latest_df['datetime'].iloc[-1]
                logger.debug(f"Data range: {first_time} to {last_time} ({len(latest_df)} rows)")

            # Filter out incomplete candles
            complete_df = latest_df[latest_df['datetime'].apply(
                lambda x: self.is_candle_complete(interval, x)
            )]

            if complete_df.empty:
                candles_checked = min(5, len(latest_df))
                candle_times = latest_df['datetime'].head(candles_checked).tolist()
                logger.info(f"No complete candles available for {interval}. Latest candles: {candle_times}")
                return None

            logger.info(f"Found {len(complete_df)} complete candles for {interval}")

            filepath = os.path.join('data', f'{interval}.csv')
            redis_key = f"{symbol}_{exchange}_{interval}"
            
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])

                # Check for new data
                new_data = complete_df[~complete_df['datetime'].isin(existing_df['datetime'])]
                if new_data.empty:
                    logger.info(f"No new {interval} data to update")
                    return existing_df
                
                logger.info(f"Found {len(new_data)} new {interval} candles to add")
                
                # Remove any overlapping data and append new complete data
                existing_df = existing_df[~existing_df['datetime'].isin(complete_df['datetime'])]
                combined_df = pd.concat([existing_df, complete_df]).sort_values('datetime')

                combined_df.to_csv(filepath, index=False)
                
                # Save to Redis
                self.redis_store.save_data(redis_key, combined_df)
                
                logger.info(f"Updated {interval} data in {filepath} and Redis (now {len(combined_df)} rows)")
                return combined_df
            else:
                complete_df.to_csv(filepath, index=False)
                
                # Save to Redis
                self.redis_store.save_data(redis_key, complete_df)
                
                logger.info(f"Created new file {filepath} and Redis entry with {interval} data ({len(complete_df)} rows)")
                return complete_df
                
        except Exception as e:
            logger.error(f"Error updating {interval} data: {e}")
            return None

if __name__ == "__main__":
    try:
        logger.info("Starting TradingView Data Fetcher")
        
        # Get login credentials
        use_credentials = input("Do you want to use TradingView credentials? (y/n): ").strip().lower() == 'y'
        username = None
        password = None
        
        if use_credentials:
            username = input("Enter your TradingView username: ")
            password = input("Enter your TradingView password: ")
            logger.info("Credentials provided, attempting authenticated connection")
        
        # Initialize the fetcher with credentials if provided
        fetcher = TradingViewDataFetcher(username, password)
        
        # Get symbol and exchange from input
        symbol = input("Enter the symbol to track (e.g., BTCUSDT): ")
        exchange = input("Enter the exchange (e.g., BINANCE): ")
        
        # First fetch historical data
        h1_data = fetcher.fetch_historical_data(symbol, exchange, 'H1', days_back=7)
        m5_data = fetcher.fetch_historical_data(symbol, exchange, 'M5', days_back=7)
        
        if h1_data is None and m5_data is None:
            logger.error("Failed to fetch initial historical data. Please check connection and retry.")
            exit(1)
        
        logger.info("\nData fetcher is now running with 15-second updates. Press Ctrl+C to stop.")
        
        update_count = 0
        last_successful_h1 = None
        last_successful_m5 = None
        
        # Run continuous updates
        while True:
            try:
                h1_updated = fetcher.update_data(symbol, exchange, 'H1')
                if h1_updated is not None:
                    last_successful_h1 = datetime.now()
                    logger.info(f"Successfully updated H1 data")
                
                m5_updated = fetcher.update_data(symbol, exchange, 'M5')
                if m5_updated is not None:
                    last_successful_m5 = datetime.now()
                    logger.info(f"Successfully updated M5 data")
                
                update_count += 1
                if update_count % 20 == 0:  # Log every 20 updates (approx. 5 minutes)
                    logger.info(f"Fetcher running for {update_count} update cycles")
                    
                    # Report time since last successful updates
                    current_time = datetime.now()
                    if last_successful_h1:
                        time_since_h1 = current_time - last_successful_h1
                        logger.info(f"Time since last successful H1 update: {time_since_h1}")
                    if last_successful_m5:
                        time_since_m5 = current_time - last_successful_m5
                        logger.info(f"Time since last successful M5 update: {time_since_m5}")
                
                time.sleep(15)  # Wait for 15 seconds before next update
                
            except KeyboardInterrupt:
                logger.info("\nData fetcher stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error during update: {e}")
                logger.info("Will retry in 15 seconds...")
                time.sleep(15)  # Wait before retrying
                
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        exit(1)