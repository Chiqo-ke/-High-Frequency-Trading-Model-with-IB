import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json
import matplotlib.pyplot as plt
import pandas_ta as ta
import numpy as np
import logging
from tvDatafeed import TvDatafeed, Interval
from functools import wraps
import random

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

def retry_on_timeout(max_retries=5, initial_delay=3, max_delay=60):
    """Enhanced decorator to retry a function on timeout with exponential backoff and jitter"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    error_msg = str(e).lower()
                    
                    if 'timed out' in error_msg or 'timeout' in error_msg:
                        # Add jitter to avoid thundering herd
                        jitter = random.uniform(0, 0.1 * delay)
                        wait_time = min(delay + jitter, max_delay)
                        
                        logger.warning(
                            f"Timeout on attempt {retries}/{max_retries}. "
                            f"Retrying in {wait_time:.1f} seconds... "
                            f"Error: {str(e)}"
                        )
                        
                        time.sleep(wait_time)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Non-timeout error occurred: {str(e)}")
                        raise

            logger.error(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            return None
        return wrapper
    return decorator

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
            
            # Ensure output directory exists
            os.makedirs('data', exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to connect to TradingView: {e}")
            raise
        self.session_start = time.time()
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

    @retry_on_timeout(max_retries=5, initial_delay=3, max_delay=60)
    def fetch_historical_data(self, symbol, exchange, interval, days_back=7):
        """Fetch historical data with improved error handling"""
        logger.info(f"Fetching historical {interval} data for {symbol}...")
        
        try:
            self._rate_limit()  # Apply rate limiting
            
            # Add delay for M5 data
            if interval == 'M5':
                time.sleep(2)
            
            # Map intervals and calculate bars
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute
            }
            
            n_bars = self._calculate_required_bars(interval, days_back)
            
            # Split large requests into smaller chunks for M5 data
            if interval == 'M5' and n_bars > 1000:
                return self._fetch_large_dataset(symbol, exchange, interval, n_bars)
            
            df = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval_map[interval],
                n_bars=n_bars
            )
            
            if df is None or df.empty:
                raise Exception(f"No data returned for {symbol} on {exchange}")
            
            return self._process_and_save_data(df, interval)
            
        except Exception as e:
            logger.error(f"Error fetching {interval} data: {str(e)}")
            raise

    def _calculate_required_bars(self, interval, days_back):
        """Calculate required number of bars with safety margin"""
        seconds_per_bar = 3600 if interval == 'H1' else 300
        n_bars = int((days_back * 24 * 3600) / seconds_per_bar)
        return int(n_bars * 1.1)  # Add 10% safety margin

    def _fetch_large_dataset(self, symbol, exchange, interval, total_bars):
        """Fetch large datasets in smaller chunks"""
        chunk_size = 1000
        chunks = []
        
        for i in range(0, total_bars, chunk_size):
            logger.info(f"Fetching chunk {i//chunk_size + 1} of {(total_bars + chunk_size - 1)//chunk_size}")
            self._rate_limit()
            
            chunk = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_5_minute,
                n_bars=min(chunk_size, total_bars - i)
            )
            
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
            time.sleep(2)  # Delay between chunks
            
        if not chunks:
            return None
            
        return pd.concat(chunks).drop_duplicates()

    def fetch_historical_data(self, symbol, exchange, interval, days_back=7):
        """Fetch historical data for specified interval and days back."""
        logger.info(f"Fetching historical {interval} data for {symbol}...")
        
        try:
            # Calculate exact timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Map intervals to tvDatafeed Interval objects
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute
            }
            
            # Calculate how many bars we need based on interval
            if interval == 'H1':
                n_bars = int((end_time - start_time).total_seconds() / 3600) + 10  # Add buffer
            elif interval == 'M5':
                n_bars = int((end_time - start_time).total_seconds() / 300) + 10  # Add buffer
            
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise

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
            
            filepath = os.path.join('data', f'{interval}.csv')
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath)
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])

                # Check for new data
                new_data = latest_df[~latest_df['datetime'].isin(existing_df['datetime'])]
                if new_data.empty:
                    logger.info(f"No new {interval} data to update")
                    return existing_df
                
                logger.info(f"Found {len(new_data)} new {interval} candles to add")
                
                # Remove any overlapping data and append new complete data
                existing_df = existing_df[~existing_df['datetime'].isin(latest_df['datetime'])]
                combined_df = pd.concat([existing_df, latest_df]).sort_values('datetime')

                combined_df.to_csv(filepath, index=False)
                logger.info(f"Updated {interval} data in {filepath} (now {len(combined_df)} rows)")
                return combined_df
            else:
                latest_df.to_csv(filepath, index=False)
                logger.info(f"Created new file {filepath} with {interval} data ({len(latest_df)} rows)")
                return latest_df
                
        except Exception as e:
            logger.error(f"Error updating {interval} data: {e}")
            return None

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
            if hasattr(candle_time, 'tz_localize') and candle_time.tzinfo is not None:
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

    @retry_on_timeout(max_retries=5, delay=3)  # Increased retries, reduced initial delay
    def update_data(self, symbol, exchange, interval):
        """Update data for the specified interval"""
        logger.info(f"Updating {interval} data for {symbol}...")

        try:
            # Add backoff delay based on interval to avoid rate limits
            if interval == 'M5':
                time.sleep(2)  # Add small delay for M5 to avoid rate limits
            
            # Map intervals to tvDatafeed Interval objects
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute
            }

            # Get the latest data with adjusted n_bars based on interval
            n_bars = 5 if interval == 'H1' else 15  # Get more bars for M5
            latest_df = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval_map[interval],
                n_bars=n_bars
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

            # Update file handling
            filename = f'{interval}.csv'
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                try:
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

                    if self._save_to_csv(combined_df, filename):
                        logger.info(f"Updated {interval} data (now {len(combined_df)} rows)")
                        return combined_df
                except Exception as e:
                    logger.error(f"Error updating existing data: {e}")
                    return None
            else:
                if self._save_to_csv(complete_df, filename):
                    logger.info(f"Created new file with {len(complete_df)} rows")
                    return complete_df
                return None

        except Exception as e:
            if 'timed out' in str(e).lower():
                logger.warning(f"Timeout while updating {interval} data, will retry...")
                raise  # Let the retry decorator handle it
            else:
                logger.error(f"Error updating {interval} data: {e}")
                return None
    
    def get_next_fetch_time(self, interval):
        """Calculate the next time to fetch data based on the interval"""
        now = datetime.now()
        
        if interval == 'H1':
            # For hourly interval, get the next hour
            next_time = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
        elif interval == 'M5':
            # For 5-minute interval, get the next 5-minute mark
            current_minute = now.minute
            next_minute = ((current_minute // 5) + 1) * 5
            
            if next_minute >= 60:  # Handle rollover to next hour
                next_time = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
            else:
                next_time = now.replace(minute=next_minute, second=5, microsecond=0)
        else:
            # Default fallback
            next_time = now + timedelta(minutes=5)
            
        return next_time
    
    def get_latest_candle_time(self, interval):
        """Get the timestamp of the most recent candle in the CSV file"""
        filepath = os.path.join(self.data_dir, f'{interval}.csv')
        if not os.path.exists(filepath):
            return None
            
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                return None
                
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df['datetime'].max()
        except Exception as e:
            logger.error(f"Error reading latest candle time: {e}")
            return None

if __name__ == "__main__":
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info("Starting TradingView Data Fetcher")
            
            # Get login credentials
            use_credentials = input("Do you want to use TradingView credentials? (y/n): ").strip().lower() == 'y'
            username = None
            password = None
            
            if use_credentials:
                print("\nNote: If login fails, the fetcher will automatically fall back to non-authenticated mode.")
                print("Non-authenticated mode may have limited data access but still works for most basic symbols.\n")
                username = input("Enter your TradingView username: ")
                password = input("Enter your TradingView password: ")
                logger.info("Attempting connection with credentials...")
            
            # Initialize the fetcher with credentials if provided
            fetcher = TradingViewDataFetcher(username, password)
            
            # Get symbol and exchange from input
            symbol = input("Enter the symbol to track (e.g., BTCUSDT): ")
            exchange = input("Enter the exchange (e.g., BINANCE): ")
            
            # First fetch historical data with better error handling
            h1_data = fetcher.fetch_historical_data(symbol, exchange, 'H1', days_back=7)
            if h1_data is None:
                logger.warning("Failed to fetch H1 data, will try to continue with M5 data")
            
            # Add delay before M5 fetch
            time.sleep(5)  # Wait 5 seconds between H1 and M5 fetches
            
            try:
                m5_data = fetcher.fetch_historical_data(symbol, exchange, 'M5', days_back=7)
                if m5_data is None:
                    logger.error("Failed to fetch M5 data")
                    if h1_data is None:
                        raise Exception("Failed to fetch both H1 and M5 data")
            except Exception as e:
                logger.error(f"Error fetching M5 data: {e}")
                if h1_data is None:
                    raise Exception("Failed to fetch both H1 and M5 data")
            
            logger.info("\nData fetcher is now running in synchronized mode. Press Ctrl+C to stop.")
            
            # Schedule next fetch times
            next_h1_fetch = fetcher.get_next_fetch_time('H1')
            next_m5_fetch = fetcher.get_next_fetch_time('M5')
            
            logger.info(f"Next H1 fetch scheduled for: {next_h1_fetch}")
            logger.info(f"Next M5 fetch scheduled for: {next_m5_fetch}")
            
            update_count = 0
            
            # Run continuous updates with synchronized timing
            while True:
                try:
                    current_time = datetime.now()
                    
                    # Add dynamic delay for frequent updates
                    update_delay = 1 if update_count < 100 else 2  # Increase delay after 100 updates
                    
                    # Check if it's time to fetch H1 data
                    if current_time >= next_h1_fetch:
                        logger.info(f"Scheduled H1 fetch time reached ({next_h1_fetch})")
                        h1_updated = fetcher.update_data(symbol, exchange, 'H1')
                        
                        if h1_updated is not None:
                            latest_h1 = fetcher.get_latest_candle_time('H1')
                            logger.info(f"Successfully updated H1 data. Latest candle: {latest_h1}")
                        
                        # Schedule next H1 fetch
                        next_h1_fetch = fetcher.get_next_fetch_time('H1')
                        logger.info(f"Next H1 fetch scheduled for: {next_h1_fetch}")
                        time.sleep(update_delay)  # Add delay between operations
                        
                    # Check if it's time to fetch M5 data
                    if current_time >= next_m5_fetch:
                        logger.info(f"Scheduled M5 fetch time reached ({next_m5_fetch})")
                        m5_updated = fetcher.update_data(symbol, exchange, 'M5')
                        
                        if m5_updated is not None:
                            latest_m5 = fetcher.get_latest_candle_time('M5')
                            logger.info(f"Successfully updated M5 data. Latest candle: {latest_m5}")
                        
                        # Schedule next M5 fetch
                        next_m5_fetch = fetcher.get_next_fetch_time('M5')
                        logger.info(f"Next M5 fetch scheduled for: {next_m5_fetch}")
                        time.sleep(update_delay)  # Add delay between operations
                    
                    update_count += 1
                    if update_count % 20 == 0:  # Log status periodically
                        logger.info(f"Fetcher running for {update_count} update cycles")
                        logger.info(f"Current time: {current_time}")
                        logger.info(f"Next H1 fetch: {next_h1_fetch} (in {(next_h1_fetch - current_time).total_seconds():.1f} seconds)")
                        logger.info(f"Next M5 fetch: {next_m5_fetch} (in {(next_m5_fetch - current_time).total_seconds():.1f} seconds)")
                    
                    # Sleep a short time to avoid CPU spinning, but stay responsive
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("\nData fetcher stopped by user.")
                    break
                except Exception as e:
                    logger.error(f"Error during update: {e}")
                    logger.info("Will retry in 15 seconds...")
                    time.sleep(15)  # Wait before retrying
                    
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 10 * retry_count
                logger.error(f"Attempt {retry_count}/{max_retries} failed: {str(e)}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.critical(f"Critical error after {max_retries} attempts: {str(e)}")
                exit(1)