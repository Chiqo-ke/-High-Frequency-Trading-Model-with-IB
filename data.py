from tvDatafeed import TvDatafeedLive, Interval
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_last_completed_m5_datetime(current_time):
    """Get the last completed 5-minute interval datetime"""
    minutes = current_time.minute
    completed_interval = (minutes // 5) * 5
    last_interval = current_time.replace(minute=completed_interval, second=0, microsecond=0)
    if minutes % 5 == 0:  # If exactly on 5-min mark, get previous interval
        last_interval -= timedelta(minutes=5)
    return last_interval

def get_last_completed_h1_datetime(current_time):
    """Get the last completed hourly interval datetime"""
    return current_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

def reconnect_tv():
    """Reconnect to TradingView"""
    try:
        logger.info("Attempting to reconnect to TradingView...")
        tv = TvDatafeedLive(username='chiqo-254', password='ChiqoMehum8844')
        logger.info("Successfully reconnected to TradingView")
        return tv
    except Exception as e:
        logger.error(f"Failed to reconnect: {str(e)}", exc_info=True)
        return None

def get_data_with_retry(tv, symbol, interval, n_bars=1):
    """Try to get data from different exchanges with retry logic"""
    exchanges = ['FX_IDC', 'OANDA']
    max_retries = len(exchanges)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            exchange = exchanges[retry_count]
            logger.info(f"Attempting to fetch data from {exchange}")
            data = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=n_bars
            )
            
            if data is not None and not data.empty:
                logger.info(f"Successfully retrieved {len(data)} bars from {exchange}")
                data = data.sort_index()
                logger.debug(f"Last 3 rows of retrieved data:\n{data.tail(3)}")
                return data
            
        except Exception as e:
            logger.error(f"Error fetching from {exchange}: {str(e)}")
        
        retry_count += 1
        if retry_count < max_retries:
            logger.info(f"Waiting 5 seconds before trying next exchange...")
            time.sleep(5)
    
    logger.error("Failed to fetch data from all exchanges")
    return None

def validate_data(df, interval_type='5m'):
    """Validate data against completion time cutoff"""
    if df.empty:
        return df
    
    current_time = datetime.now()
    
    # Get appropriate cutoff time based on interval
    if interval_type == '5m':
        cutoff = get_last_completed_m5_datetime(current_time)
    elif interval_type == '1h':
        cutoff = get_last_completed_h1_datetime(current_time)
    else:
        logger.error(f"Unsupported interval type: {interval_type}")
        return pd.DataFrame()
    
    # Filter out any future-dated or incomplete data
    valid_data = df[df.index <= cutoff]
    
    # Remove data points newer than cutoff
    removed_count = len(df) - len(valid_data)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} incomplete/future data points. "
                      f"Last valid timestamp: {cutoff}")
    
    # Additional quality checks
    if not valid_data.empty:
        # Check for missing values
        if valid_data.isnull().values.any():
            logger.error("Null values detected in validated data")
            return pd.DataFrame()
        
        # Check for duplicate timestamps
        duplicates = valid_data.index.duplicated(keep='last')
        if duplicates.any():
            logger.warning(f"Removed {duplicates.sum()} duplicate timestamps")
            valid_data = valid_data[~duplicates]
    
    return valid_data

def save_ohlcv_data(filename, new_data, existing_data=None):
    """Save only OHLCV data to CSV"""
    try:
        # Load existing data if not provided
        if existing_data is None and os.path.exists(filename):
            existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        if existing_data is not None and not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        else:
            combined_data = new_data
            
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        combined_data[ohlcv_columns].sort_index().to_csv(filename, index=True)
        logger.info(f"Saved {len(new_data)} new bars to {filename} (total: {len(combined_data)})")
        return combined_data
    except Exception as e:
        logger.error(f"Error saving OHLCV data to {filename}: {str(e)}")
        return None