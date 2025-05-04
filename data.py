
import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
from concurrent.futures import ThreadPoolExecutor
import configparser
import signal
import sys

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
    # Constants
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 5  # seconds
    INTERVALS = {
        'H1': Interval.in_1_hour,
        'M5': Interval.in_5_minute,
        # Add more intervals if needed
    }
    
    # Interval durations in seconds - used for timing calculations
    INTERVAL_DURATIONS = {
        'H1': 3600,  # 1 hour in seconds
        'M5': 300,   # 5 minutes in seconds
    }
    
    def __init__(self, username=None, password=None, data_dir='data'):
        """Initialize the TradingView data fetcher with optional credentials."""
        self.username = username
        self.password = password
        self.data_dir = data_dir
        self.symbol = None
        self.exchange = None
        self.running = False
        self.last_successful_update = {}
        self.last_attempt_update = {}  # NEW: Track last attempt time for each interval
        
        # Ensure output directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Connect to TradingView
        self._connect()
        
    def _connect(self):
        """Establish connection to TradingView with error handling."""
        logger.info("Connecting to TradingView...")
        
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                if self.username and self.password:
                    self.tv = TvDatafeed(username=self.username, password=self.password)
                    logger.info("Connection established with credentials!")
                else:
                    self.tv = TvDatafeed()
                    logger.info("Connection established without login (some data may be limited)")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt+1}/{self.RETRY_ATTEMPTS} failed: {e}")
                if attempt < self.RETRY_ATTEMPTS - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.error("Failed to connect to TradingView after multiple attempts")
                    raise
        
    def _get_filepath(self, interval):
        """Get standardized filepath for a given interval."""
        return os.path.join(self.data_dir, f'{self.symbol}_{self.exchange}_{interval}.csv')
        
    def fetch_historical_data(self, symbol, exchange, interval, days_back=7):
        """Fetch historical data for specified interval and days back, excluding incomplete candles."""
        self.symbol = symbol
        self.exchange = exchange
        
        logger.info(f"Fetching historical {interval} data for {symbol} on {exchange}...")
        
        try:
            # Calculate exact timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Determine how many bars to fetch based on interval
            interval_seconds = self.INTERVAL_DURATIONS.get(interval, 300)
            n_bars = int((end_time - start_time).total_seconds() / interval_seconds) + 50  # Increased buffer
            
            logger.debug(f"Requesting {n_bars} bars for {interval}")
            
            # Fetch the data
            df = self._fetch_with_retry(symbol, exchange, interval, n_bars)
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} on {exchange} with interval {interval}")
                return None
                
            # Reset index to make datetime a column and format for CSV
            df = df.reset_index()
            
            # Filter out incomplete candles
            logger.info(f"Filtering out incomplete candles from historical data")
            orig_len = len(df)
            df = df[df['datetime'].apply(lambda x: self.is_candle_complete(interval, x))]
            logger.info(f"Removed {orig_len - len(df)} incomplete candles from historical data")
            
            if df.empty:
                logger.warning("All fetched historical candles are incomplete. Try increasing days_back.")
                return None
            
            # Save to CSV
            filepath = self._get_filepath(interval)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved historical data to {filepath} ({len(df)} complete candles)")
            
            # Log the time range of the data
            start_date = df['datetime'].min()
            end_date = df['datetime'].max()
            logger.info(f"Historical data range: {start_date} to {end_date}")
            
            # Store the last successful update time
            self.last_successful_update[interval] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def _fetch_with_retry(self, symbol, exchange, interval, n_bars):
        """Fetch data with retry logic."""
        for attempt in range(self.RETRY_ATTEMPTS):
            try:
                return self.tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=self.INTERVALS[interval],
                    n_bars=n_bars
                )
            except Exception as e:
                logger.error(f"Fetch attempt {attempt+1}/{self.RETRY_ATTEMPTS} failed: {e}")
                if attempt < self.RETRY_ATTEMPTS - 1:
                    logger.info(f"Retrying in {self.RETRY_DELAY} seconds...")
                    time.sleep(self.RETRY_DELAY)
        
        logger.error(f"Failed to fetch data after {self.RETRY_ATTEMPTS} attempts")
        return None
    
    def is_candle_complete(self, interval, datetime_val):
        """Check if the current candle is complete."""
        try:
            current_time = datetime.now()
            
            # Convert to datetime if string
            if isinstance(datetime_val, str):
                candle_time = pd.to_datetime(datetime_val)
            else:
                candle_time = datetime_val
                
            # Remove timezone info if present
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

            # Check if candle is complete - strictly require start of next candle
            is_complete = current_time >= next_candle
            
            # Add a larger buffer for H1 candles to ensure candle is really complete
            if interval == 'H1':
                buffer = pd.Timedelta(seconds=120)  # 2 minutes buffer for hourly candles
            elif interval == 'M5':
                buffer = pd.Timedelta(seconds=10)   # 10 seconds buffer for 5-min candles
                
            is_complete_with_buffer = current_time >= (next_candle + buffer)
            
            if is_complete and not is_complete_with_buffer:
                logger.debug(f"Candle at {candle_time} is technically complete but still in buffer period")
                return False  # Still in buffer period, consider incomplete
                
            return is_complete
            
        except Exception as e:
            logger.error(f"Error checking if candle is complete: {e}")
            return False

    def update_data(self, symbol, exchange, interval):
        """Update data for the specified interval, ignoring currently open candles."""
        if not self.symbol:
            self.symbol = symbol
            self.exchange = exchange
            
        logger.debug(f"Updating {interval} data for {symbol}...")
        
        # Record attempt time
        self.last_attempt_update[interval] = datetime.now()

        try:
            # Get the latest data with extra bars for better context
            # Get more bars for H1 to ensure we have sufficient history
            n_bars = 50 if interval == 'H1' else 30
            latest_df = self._fetch_with_retry(symbol, exchange, interval, n_bars)
            
            if latest_df is None or latest_df.empty:
                logger.warning(f"No data returned when updating {interval} data")
                return None

            latest_df = latest_df.reset_index()
            
            # Log the most recent candle time before filtering
            if not latest_df.empty:
                most_recent = latest_df['datetime'].max()
                logger.debug(f"Most recent {interval} candle before filtering: {most_recent}")
            
            # Strictly filter out incomplete candles
            complete_df = latest_df[latest_df['datetime'].apply(
                lambda x: self.is_candle_complete(interval, x)
            )]

            if complete_df.empty:
                logger.debug(f"No complete candles available for {interval} - current candle still in progress")
                return None
                
            # Log the most recent complete candle
            most_recent_complete = complete_df['datetime'].max()
            logger.debug(f"Most recent complete {interval} candle: {most_recent_complete}")
            
            # Calculate difference from current time
            time_diff = datetime.now() - pd.to_datetime(most_recent_complete)
            logger.debug(f"Current complete candle is {time_diff} old")

            # Process file update
            filepath = self._get_filepath(interval)
            
            if os.path.exists(filepath):
                # Update existing file
                existing_df = pd.read_csv(filepath)
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])

                # Check for new data - using strict equality check
                existing_datetimes = set(existing_df['datetime'])
                new_data = complete_df[~complete_df['datetime'].isin(existing_datetimes)]
                
                if not new_data.empty:
                    # Log details about new data
                    new_dates = sorted(new_data['datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist())
                    logger.info(f"Found {len(new_data)} new {interval} candles: {new_dates}")
                    
                    # Remove any overlapping data and append new complete data
                    # Use pandas merge to handle this efficiently
                    combined_df = pd.concat([
                        existing_df[~existing_df['datetime'].isin(complete_df['datetime'])], 
                        complete_df
                    ])
                    # Sort to ensure chronological order
                    combined_df = combined_df.sort_values('datetime')
                    
                    # Save updated data
                    combined_df.to_csv(filepath, index=False)
                    logger.info(f"Updated {interval} data with {len(new_data)} new complete candles")
                    
                    # Update last successful time
                    self.last_successful_update[interval] = datetime.now()
                    return combined_df
                else:
                    logger.debug(f"No new complete {interval} candles to add")
                    return existing_df
            else:
                # Create new file with only complete candles
                complete_df.to_csv(filepath, index=False)
                logger.info(f"Created new file {filepath} with {len(complete_df)} complete candles")
                
                # Update last successful time
                self.last_successful_update[interval] = datetime.now()
                return complete_df
                
        except Exception as e:
            logger.error(f"Error updating {interval} data: {e}")
            return None
    
    def _calculate_next_check_time(self, interval):
        """Calculate the next time to check for a completed candle based on interval."""
        current_time = datetime.now()
        
        if interval == 'H1':
            # For hourly candles, don't wait for the start of the next hour
            # Instead, check every 15 minutes to make sure we don't miss updates
            minutes_to_add = 15
            next_check = current_time + timedelta(minutes=minutes_to_add)
            return next_check
        elif interval == 'M5':
            # For 5-minute candles, check at next 5-minute mark
            current_minutes = current_time.minute
            # Calculate minutes until next 5-minute mark
            minutes_to_add = 5 - (current_minutes % 5)
            if minutes_to_add == 0:  # If exactly at 5-minute mark, wait for next one
                minutes_to_add = 5
            
            next_check = current_time.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
            # Add a small buffer to ensure the candle is complete
            return next_check + timedelta(seconds=10)
        else:
            # For unknown intervals, default to checking in 1 minute
            logger.warning(f"Unknown interval for check timing: {interval}")
            return current_time + timedelta(minutes=1)
            
    def _update_worker(self, symbol, exchange, intervals):
        """Worker function to update data for multiple intervals with smart timing."""
        # Track next check time for each interval
        next_check_times = {interval: datetime.now() for interval in intervals}
        
        # Force initial immediate check for H1 data
        logger.info("Performing initial check for all intervals")
        for interval in intervals:
            self.update_data(symbol, exchange, interval)
        
        while self.running:
            try:
                current_time = datetime.now()
                intervals_to_check = []
                
                # Determine which intervals need checking now
                for interval in intervals:
                    if current_time >= next_check_times[interval]:
                        intervals_to_check.append(interval)
                        # Calculate the next check time
                        next_check_times[interval] = self._calculate_next_check_time(interval)
                        logger.debug(f"Scheduled next {interval} check at {next_check_times[interval]}")
                
                # Only update intervals that need checking
                if intervals_to_check:
                    logger.info(f"Checking intervals: {intervals_to_check}")
                    
                    # Use ThreadPoolExecutor for concurrent updates
                    with ThreadPoolExecutor(max_workers=len(intervals_to_check)) as executor:
                        futures = {
                            interval: executor.submit(self.update_data, symbol, exchange, interval)
                            for interval in intervals_to_check
                        }
                        
                        # Process results
                        for interval, future in futures.items():
                            try:
                                result = future.result()
                                if result is not None:
                                    logger.info(f"Successfully updated {interval} data")
                            except Exception as e:
                                logger.error(f"Error updating {interval} data in worker: {e}")
                
                # Additional check: If no updates for H1 in a long time, force check
                for interval in intervals:
                    if interval == 'H1':
                        last_success = self.last_successful_update.get(interval)
                        if (last_success is not None and 
                            (datetime.now() - last_success).total_seconds() > 7200):  # 2 hours
                            # Force a check if it's been more than 2 hours since update
                            logger.warning(f"No {interval} updates for 2+ hours, forcing check")
                            try:
                                self.update_data(symbol, exchange, interval)
                                # Recalculate next check time regardless of result
                                next_check_times[interval] = self._calculate_next_check_time(interval)
                            except Exception as e:
                                logger.error(f"Error in forced {interval} update: {e}")
                
                # Calculate sleep time until next check
                if next_check_times:
                    next_check = min(next_check_times.values())
                    sleep_time = (next_check - datetime.now()).total_seconds()
                    
                    # Ensure sleep time is reasonable
                    if sleep_time > 0:
                        logger.debug(f"Sleeping for {sleep_time:.1f} seconds until next check")
                        time.sleep(min(sleep_time, 60))  # Cap at 60 seconds as a safety
                    else:
                        time.sleep(1)  # Minimal sleep to prevent CPU hogging
                else:
                    time.sleep(30)  # Default sleep if no intervals
                    
            except Exception as e:
                logger.error(f"Error in update worker: {e}")
                time.sleep(5)  # Short delay before retry
    
    def start_continuous_updates(self, symbol, exchange, intervals=None):
        """Start continuous updates for specified intervals."""
        if intervals is None:
            intervals = list(self.INTERVALS.keys())
            
        self.symbol = symbol
        self.exchange = exchange
        self.running = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start update worker in the current thread
        logger.info(f"Starting continuous updates for {symbol} on {exchange}")
        self._update_worker(symbol, exchange, intervals)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info("Received shutdown signal, stopping updates...")
        self.running = False
        logger.info("Data fetcher stopped gracefully.")
        sys.exit(0)
    
    def status_report(self):
        """Generate status report."""
        current_time = datetime.now()
        report = ["--- TradingView Data Fetcher Status Report ---"]
        
        report.append(f"Symbol: {self.symbol}")
        report.append(f"Exchange: {self.exchange}")
        report.append(f"Running: {self.running}")
        
        for interval in self.INTERVALS.keys():
            # Information about last successful update
            last_time = self.last_successful_update.get(interval)
            if last_time:
                time_since = current_time - last_time
                report.append(f"{interval} last updated: {last_time} ({time_since} ago)")
            else:
                report.append(f"{interval} has never been successfully updated")
                
            # Information about last attempt
            last_attempt = self.last_attempt_update.get(interval)
            if last_attempt:
                attempt_since = current_time - last_attempt
                report.append(f"  - Last update attempt: {last_attempt} ({attempt_since} ago)")
            
            # Check file size and content
            filepath = self._get_filepath(interval)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # KB
                df = pd.read_csv(filepath)
                most_recent = pd.to_datetime(df['datetime']).max() if not df.empty else "N/A"
                report.append(f"  - File size: {file_size:.2f} KB, Rows: {len(df)}")
                report.append(f"  - Most recent candle: {most_recent}")
        
        return "\n".join(report)

def load_config(config_file='config.ini'):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_file):
        config['TradingView'] = {
            'username': '',
            'password': '',
        }
        config['Data'] = {
            'symbol': 'BTCUSDT',
            'exchange': 'BINANCE',
            'intervals': 'H1,M5',
            'days_back': '7',
        }
        
        with open(config_file, 'w') as f:
            config.write(f)
        
        logger.info(f"Created default configuration file: {config_file}")
        logger.info("Please edit the configuration file and restart the program.")
        return None
        
    # Load existing config
    config.read(config_file)
    return config

def get_user_input(prompt, default=None):
    """Get user input with default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def main():
    """Main entry point."""
    try:
        logger.info("Starting TradingView Data Fetcher")
        
        # Try to load config
        config = load_config()
        if config is None:
            return
            
        # Get configuration
        username = get_user_input("Enter your TradingView username (leave blank for guest access)", 
                                 config['TradingView'].get('username', ''))
        
        password = None
        if username:
            password = get_user_input("Enter your TradingView password", 
                                     config['TradingView'].get('password', ''))
            
        symbol = get_user_input("Enter the symbol to track", 
                               config['Data'].get('symbol', 'BTCUSDT'))
                               
        exchange = get_user_input("Enter the exchange", 
                                 config['Data'].get('exchange', 'BINANCE'))
                                 
        intervals_str = get_user_input("Enter comma-separated intervals to track (H1,M5)", 
                                      config['Data'].get('intervals', 'H1,M5'))
        intervals = [i.strip() for i in intervals_str.split(',')]
        
        days_back = int(get_user_input("Enter number of days of historical data to fetch", 
                                      config['Data'].get('days_back', '7')))
        
        # Initialize the fetcher
        fetcher = TradingViewDataFetcher(username if username else None, 
                                        password if password else None)
        
        # Fetch historical data for each interval
        for interval in intervals:
            if interval not in fetcher.INTERVALS:
                logger.error(f"Invalid interval: {interval}")
                continue
                
            fetcher.fetch_historical_data(symbol, exchange, interval, days_back=days_back)
        
        # Start continuous updates
        logger.info("\nData fetcher is now running. Press Ctrl+C to stop.")
        fetcher.start_continuous_updates(symbol, exchange, intervals)
        
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    main()