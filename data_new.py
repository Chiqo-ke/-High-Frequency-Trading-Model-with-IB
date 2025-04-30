import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging
from tvDatafeed import TvDatafeed, Interval
import requests
from requests.exceptions import Timeout, RequestException
from functools import wraps
import json
from json.decoder import JSONDecodeError

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

def validate_response(response_data):
    """Validate API response before JSON parsing"""
    if not response_data or response_data.strip() == '':
        raise ValueError("Empty response received from TradingView API")
    
    # Check if response starts with valid JSON character
    if not response_data.strip().startswith('{') and not response_data.strip().startswith('['):
        raise ValueError(f"Invalid JSON response: {response_data[:100]}...")
    return response_data

def format_symbol_for_exchange(symbol, exchange):
    """Format symbol according to exchange requirements"""
    if exchange.upper() == "OANDA":
        # OANDA often requires underscores between currency pairs
        if "_" not in symbol and len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}"
    return symbol

def retry_on_exception(max_retries=3, delay=5, backoff_factor=2, exceptions=(Timeout, RequestException, JSONDecodeError)):
    """Retry decorator with exponential backoff for network-related exceptions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Final error: {e}")
                        raise
                    
                    # Check for rate limiting indicators
                    if "rate" in str(e).lower() or "limit" in str(e).lower() or "too many" in str(e).lower():
                        logger.warning("Possible rate limiting detected - increasing backoff")
                        current_delay *= 2  # Double the delay for rate limiting
                    
                    logger.warning(f"Exception occurred: {type(e).__name__}: {e}. Retry {retries}/{max_retries} after {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor  # Exponential backoff
            
            return None
        return wrapper
    return decorator

class TradingViewDataFetcher:
    def __init__(self, username=None, password=None, connection_timeout=30):
        """Initialize the TradingView data fetcher with optional credentials."""
        logger.info("Connecting to TradingView...")
        self.connection_timeout = connection_timeout
        
        # Add retry logic to the initialization
        for attempt in range(3):
            try:
                # Use a longer timeout for OANDA connections
                if username and password:
                    self.tv = TvDatafeed(username=username, password=password, timeout=60)
                else:
                    self.tv = TvDatafeed(timeout=60)
                    
                logger.info("Connection established" + (" with credentials!" if username and password else " without login (some data may be limited)"))
                
                # Ensure output directory exists
                os.makedirs('data', exist_ok=True)
                break  # Connection successful, exit retry loop
                
            except Exception as e:
                if attempt < 2:  # Try 3 times (0, 1, 2)
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Connection attempt {attempt+1} failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to TradingView after 3 attempts: {e}")
                    raise

    @retry_on_exception(max_retries=5, delay=10, backoff_factor=1.5)
    def fetch_historical_data(self, symbol, exchange, interval, days_back=7):
        """Fetch historical data for specified interval and days back."""
        logger.info(f"Fetching historical {interval} data for {symbol} from {exchange}...")
        
        try:
            # Format the symbol properly for the exchange
            formatted_symbol = format_symbol_for_exchange(symbol, exchange)
            if formatted_symbol != symbol:
                logger.info(f"Using formatted symbol {formatted_symbol} for {exchange}")

            # Calculate exact timestamps
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Add more detailed logging
            logger.info(f"Fetching data from {start_time} to {end_time}")
            
            # Map our interval names to tvDatafeed Interval objects
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute,
            }
            
            if interval not in interval_map:
                logger.error(f"Unsupported interval: {interval}")
                return None
            
            # Calculate how many bars we need based on interval
            seconds_per_bar = 3600 if interval == 'H1' else 300
            n_bars = int((end_time - start_time).total_seconds() / seconds_per_bar) + 20  # Add more buffer
            
            logger.debug(f"Requesting {n_bars} bars for {interval}")
            
            # Check if exchange supports the symbol
            exchanges = self.tv.search_symbol(symbol)
            if exchanges and exchange.upper() not in [ex.upper() for ex in exchanges.keys()]:
                available = ", ".join(exchanges.keys())
                logger.warning(f"Exchange {exchange} may not support {symbol}. Available exchanges: {available}")
            
            # Handle OANDA specific parameters
            kwargs = {'extended_session': False}
            if exchange.upper() == "OANDA":
                kwargs['timeout'] = 90  # Longer timeout for OANDA

            # Fetch the data with timeout handling
            try:
                df = self.tv.get_hist(
                    symbol=formatted_symbol,
                    exchange=exchange,
                    interval=interval_map[interval],
                    n_bars=n_bars,
                    **kwargs
                )
            except JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}. TradingView API may be rate limiting your requests.")
                time.sleep(15)  # Wait before retry
                raise Timeout("Rate limited by TradingView API")
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol} on {exchange} with interval {interval}")
                return None
                
            # Reset index to make datetime a column
            df = df.reset_index()
            
            # Save to CSV
            filepath = os.path.join('data', f'{interval}.csv')
            df.to_csv(filepath, index=False)
            logger.info(f"Saved historical data to {filepath} ({len(df)} rows)")
            
            return df
            
        except Timeout as e:
            logger.error(f"Timeout while fetching data: {e}")
            raise
        except RequestException as e:
            logger.error(f"Request exception while fetching data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            if "not found" in str(e).lower():
                logger.error(f"Symbol {symbol} may not exist on exchange {exchange}")
            raise
    
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
            if pd.notnull(candle_time) and hasattr(candle_time, 'tz_localize'):
                candle_time = candle_time.tz_localize(None)
            
            # Calculate when the next candle should start
            delta = pd.Timedelta(hours=1) if interval == 'H1' else pd.Timedelta(minutes=5)
            next_candle = candle_time + delta
            
            # Debug information
            logger.debug(f"Candle time: {candle_time}, Next candle: {next_candle}, Current time: {current_time}")
            return current_time >= next_candle
            
        except Exception as e:
            logger.error(f"Error checking if candle is complete: {e}")
            return False

    @retry_on_exception(max_retries=3, delay=5)
    def update_data(self, symbol, exchange, interval, lookback_bars=20):
        """Update data for the specified interval"""
        logger.info(f"Updating {interval} data for {symbol}...")

        try:
            # Map intervals to tvDatafeed Interval objects
            interval_map = {
                'H1': Interval.in_1_hour,
                'M5': Interval.in_5_minute,
                # Add other intervals if needed
            }

            if interval not in interval_map:
                logger.error(f"Unsupported interval: {interval}")
                return None

            # Get more historical bars to ensure we catch completed orders
            latest_df = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval_map[interval],
                n_bars=lookback_bars
            )
            
            if latest_df is None or latest_df.empty:
                logger.warning(f"No data returned when updating {interval} data")
                return None

            latest_df = latest_df.reset_index()
            
            # Store last known complete candle time
            filepath = os.path.join('data', f'{interval}_last_complete.txt')
            last_complete_time = None
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    last_complete_time = pd.to_datetime(f.read().strip())

            # Filter out incomplete candles
            complete_df = latest_df[latest_df['datetime'].apply(
                lambda x: self.is_candle_complete(interval, x)
            )]

            if not complete_df.empty:
                # Save the most recent complete candle time
                with open(filepath, 'w') as f:
                    f.write(str(complete_df['datetime'].max()))

            if complete_df.empty:
                candles_checked = min(5, len(latest_df))
                candle_times = latest_df['datetime'].head(candles_checked).tolist()
                logger.info(f"No complete candles available for {interval}. Latest candles: {candle_times}")
                return None

            logger.info(f"Found {len(complete_df)} complete candles for {interval}")

            filepath = os.path.join('data', f'{interval}.csv')
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
                logger.info(f"Updated {interval} data in {filepath} (now {len(combined_df)} rows)")
                return combined_df
            else:
                complete_df.to_csv(filepath, index=False)
                logger.info(f"Created new file {filepath} with {interval} data ({len(complete_df)} rows)")
                return complete_df
                
        except Exception as e:
            logger.error(f"Error updating {interval} data: {e}")
            raise  # Re-raise to trigger retry


def symbol_exists(tv, symbol, exchange):
    """Check if a symbol exists on a given exchange"""
    try:
        search_results = tv.search_symbol(symbol)
        if not search_results:
            return False
        
        for ex, symbols in search_results.items():
            if ex.upper() == exchange.upper():
                return True
        
        logger.warning(f"Symbol {symbol} not found on {exchange}. Available exchanges: {list(search_results.keys())}")
        return False
    except Exception as e:
        logger.error(f"Error checking symbol existence: {e}")
        return False


if __name__ == "__main__":
    try:
        logger.info("Starting TradingView Data Fetcher")
        
        # Get login credentials
        use_credentials = input("Do you want to use TradingView credentials? (y/n): ").strip().lower() == 'y'
        username = password = None
        
        if use_credentials:
            username = input("Enter your TradingView username: ")
            password = input("Enter your TradingView password: ")
            logger.info("Credentials provided, attempting authenticated connection")
        
        # Initialize the fetcher with credentials if provided
        fetcher = TradingViewDataFetcher(username, password)
        
        # Get symbol and exchange from input
        symbol = input("Enter the symbol to track (e.g., BTCUSDT): ")
        exchange = input("Enter the exchange (e.g., BINANCE): ")

        # Format symbol for the specific exchange
        original_symbol = symbol
        symbol = format_symbol_for_exchange(symbol, exchange)
        if symbol != original_symbol:
            logger.info(f"Formatted {original_symbol} to {symbol} for {exchange}")
        
        # First validate the symbol exists on the exchange
        if not symbol_exists(fetcher.tv, symbol, exchange):
            logger.warning(f"Symbol {symbol} may not exist on {exchange} or requires different formatting.")
            alt_symbol = input(f"Try an alternative symbol or press Enter to continue with {symbol}: ")
            if alt_symbol:
                symbol = alt_symbol
        
        # First fetch historical data with more robust error handling
        h1_data = m5_data = None
        
        try:
            h1_data = fetcher.fetch_historical_data(symbol, exchange, 'H1', days_back=7)
            logger.info("Waiting 15 seconds to avoid rate limiting...")
            time.sleep(15)  # Add delay between requests
            m5_data = fetcher.fetch_historical_data(symbol, exchange, 'M5', days_back=7)
        except Exception as e:
            logger.error(f"Failed to fetch initial data: {e}")
        
        if h1_data is None and m5_data is None:
            logger.error("Failed to fetch any initial historical data.")
            retry = input("Do you want to retry with different parameters? (y/n): ").strip().lower()
            if retry == 'y':
                symbol = input("Enter a different symbol: ")
                exchange = input("Enter a different exchange: ")
                h1_data = fetcher.fetch_historical_data(symbol, exchange, 'H1', days_back=7)
                m5_data = fetcher.fetch_historical_data(symbol, exchange, 'M5', days_back=7)
                
                if h1_data is None and m5_data is None:
                    logger.error("Still failed to fetch data. Exiting.")
                    exit(1)
            else:
                logger.error("Exiting due to data fetch failure.")
                exit(1)
        
        logger.info("\nData fetcher is now running with 15-second updates. Press Ctrl+C to stop.")
        
        update_count = 0
        consecutive_failures = 0
        last_successful_h1 = None
        last_successful_m5 = None
        
        # Run continuous updates
        while True:
            try:
                h1_updated = None
                m5_updated = None
                
                try:
                    h1_updated = fetcher.update_data(symbol, exchange, 'H1')
                    if h1_updated is not None:
                        last_successful_h1 = datetime.now()
                        logger.info(f"Successfully updated H1 data")
                        consecutive_failures = 0
                except Exception as e:
                    logger.error(f"Error updating H1 data: {e}")
                
                try:
                    m5_updated = fetcher.update_data(symbol, exchange, 'M5')
                    if m5_updated is not None:
                        last_successful_m5 = datetime.now()
                        logger.info(f"Successfully updated M5 data")
                        consecutive_failures = 0
                except Exception as e:
                    logger.error(f"Error updating M5 data: {e}")
                
                if h1_updated is None and m5_updated is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        logger.warning(f"10 consecutive update failures. There may be a connection issue or rate limiting.")
                        consecutive_failures = 0  # Reset counter
                        time.sleep(60)  # Take a longer break
                        continue
                
                update_count += 1
                if update_count % 20 == 0:  # Log every 20 updates (approx. 5 minutes)
                    logger.info(f"Fetcher running for {update_count} update cycles")
                    
                    # Report time since last successful updates
                    current_time = datetime.now()
                    if last_successful_h1:
                        logger.info(f"Time since last successful H1 update: {current_time - last_successful_h1}")
                    if last_successful_m5:
                        logger.info(f"Time since last successful M5 update: {current_time - last_successful_m5}")
                
                time.sleep(15)  # Wait for 15 seconds before next update
                
            except KeyboardInterrupt:
                logger.info("\nData fetcher stopped by user.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during update cycle: {e}")
                logger.info("Will retry in 30 seconds...")
                time.sleep(30)  # Longer wait before retrying after unexpected error
                
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")
        logger.critical("Application is exiting. Check logs for details.")