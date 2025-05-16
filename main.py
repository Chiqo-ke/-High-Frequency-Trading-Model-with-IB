from tvDatafeed import TvDatafeedLive, Interval
from datetime import datetime, timedelta
import logging
import time
import pandas as pd
import os
from data import reconnect_tv, get_data_with_retry, validate_data, save_ohlcv_data
from signals import update_indicators
from data import (
    reconnect_tv, get_data_with_retry, validate_data, save_ohlcv_data,
    get_last_completed_m5_datetime, get_last_completed_h1_datetime
)
from signals import update_indicators

logger = logging.getLogger(__name__)

def initialize_data(tv, symbol, interval, n_bars, filename, interval_type):
    """Initialize or update data file with historical data"""
    try:
        logger.info(f"Initializing {interval_type} data...")
        data = get_data_with_retry(tv, symbol, interval, n_bars)
        if data is None:
            return None
        
        validated_data = validate_data(data, interval_type)
        if validated_data.empty:
            logger.error(f"Failed to validate initial {interval_type} data")
            return None
        
        saved_data = save_ohlcv_data(filename, validated_data)
        if saved_data is not None:
            update_indicators(filename, interval_type)
            logger.info(f"{interval_type} data initialized with {len(saved_data)} records")
        return saved_data
    except Exception as e:
        logger.error(f"Error initializing {interval_type} data: {str(e)}")
        return None

def real_time_monitor():
    """Main monitoring loop for real-time updates"""
    tv = None
    symbol = 'EURUSD'
    m5_file = 'M5.csv'
    h1_file = 'H1.csv'
    
    # Initialize connection
    tv = reconnect_tv()
    if tv is None:
        logger.critical("Failed to establish initial connection")
        return

    # Load or initialize data
    m5_data = pd.DataFrame()
    h1_data = pd.DataFrame()
    
    if os.path.exists(m5_file):
        m5_data = pd.read_csv(m5_file, index_col=0, parse_dates=True)
    if os.path.exists(h1_file):
        h1_data = pd.read_csv(h1_file, index_col=0, parse_dates=True)

    # Main loop
    while True:
        try:
            current_time = datetime.now()
            m5_cutoff = get_last_completed_m5_datetime(current_time)
            h1_cutoff = get_last_completed_h1_datetime(current_time)
            
            # Check and update M5 data
            if m5_data.empty or m5_data.index[-1] < m5_cutoff:
                logger.info("Checking for new M5 data...")
                new_m5 = get_data_with_retry(tv, symbol, Interval.in_5_minute, n_bars=10)
                if new_m5 is not None:
                    valid_m5 = validate_data(new_m5, '5m')
                    if not valid_m5.empty and valid_m5.index[-1] > m5_data.index[-1]:
                        m5_data = save_ohlcv_data(m5_file, valid_m5, m5_data)
                        update_indicators(m5_file, 'M5')
            
            # Check and update H1 data (every hour)
            if h1_data.empty or (current_time.minute == 0 and current_time.second < 30):
                logger.info("Checking for new H1 data...")
                new_h1 = get_data_with_retry(tv, symbol, Interval.in_1_hour, n_bars=24)
                if new_h1 is not None:
                    valid_h1 = validate_data(new_h1, '1h')
                    if not valid_h1.empty and valid_h1.index[-1] > h1_data.index[-1]:
                        h1_data = save_ohlcv_data(h1_file, valid_h1, h1_data)
                        update_indicators(h1_file, 'H1')
            
            # Calculate sleep time until next potential update
            next_m5 = m5_cutoff + timedelta(minutes=5)
            sleep_time = (next_m5 - datetime.now()).total_seconds() + 5  # Add buffer
            sleep_time = max(sleep_time, 30)  # Minimum 30 seconds
            
            logger.info(f"Next check in {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}", exc_info=True)
            tv = None
            time.sleep(60)
            tv = reconnect_tv()

if __name__ == '__main__':
    # Initialize historical data
    tv = reconnect_tv()
    if tv:
        # Initialize M5 data with 2 weeks history
        m5_data = initialize_data(tv, 'EURUSD', Interval.in_5_minute, 4032, 'M5.csv', '5m')
        # Initialize H1 data with 1 month history
        h1_data = initialize_data(tv, 'EURUSD', Interval.in_1_hour, 672, 'H1.csv', '1h')
        
        if m5_data is not None and h1_data is not None:
            logger.info("Starting real-time monitoring")
            real_time_monitor()
        else:
            logger.error("Failed to initialize historical data")
    else:
        logger.error("Failed to connect to TradingView")