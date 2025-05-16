from tvDatafeed import TvDatafeedLive, Interval
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from talipp.indicators import MACD, RSI, EMA
from talipp.indicator_util import composite_to_lists
import logging
import time

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

def calculate_indicators(df, timeframe='H1'):
    """Calculate indicators if data exists and has enough points"""
    if df is None or df.empty:
        logger.error("No data available for indicator calculation")
        return None
        
    close_values = df['close'].tolist()
    if len(close_values) < 17:  # Minimum points needed for MACD(8,17,9)
        logger.error(f"Not enough data points for indicators. Need at least 17, got {len(close_values)}")
        return None
    
    try:
        # Calculate MACD
        macd_ind = MACD(8, 17, 9, input_values=close_values)
        macd_result = composite_to_lists(macd_ind)
        if not macd_result:
            logger.error("MACD calculation returned no results")
            return None
        macd_vals, signal_vals, hist_vals = macd_result
        
        df['MACD_8_17_9'] = macd_vals
        df['MACDs_8_17_9'] = signal_vals
        df['MACDh_8_17_9'] = hist_vals
        
        # Calculate MACD crossovers
        df['macd_cross_below'] = np.where(
            (df['MACD_8_17_9'] < df['MACDs_8_17_9']) & 
            (df['MACD_8_17_9'].shift(1) >= df['MACDs_8_17_9'].shift(1)),
            True, False
        )
        df['macd_cross_above'] = np.where(
            (df['MACD_8_17_9'] > df['MACDs_8_17_9']) & 
            (df['MACD_8_17_9'].shift(1) <= df['MACDs_8_17_9'].shift(1)),
            True, False
        )
        
        # Calculate RSI
        rsi_ind = RSI(9, input_values=close_values)
        df['rsi_9'] = list(rsi_ind)
        
        # Calculate EMAs based on timeframe
        if timeframe == 'H1':
            ema_ind = EMA(8, input_values=close_values)
            df['ema_8'] = list(ema_ind)
            df['trend_bias'] = np.where(df['close'] > df['ema_8'], 'bullish', 'bearish')
        else:  # M5
            ema_ind = EMA(5, input_values=close_values)
            df['ema_5'] = list(ema_ind)
            df['price_crossed_above_ema'] = np.where(
                (df['close'] > df['ema_5']) & 
                (df['close'].shift(1) <= df['ema_5'].shift(1)),
                True, False
            )
            df['price_crossed_below_ema'] = np.where(
                (df['close'] < df['ema_5']) & 
                (df['close'].shift(1) >= df['ema_5'].shift(1)),
                True, False
            )
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return None

def validate_data(df):
    # Get the last row datetime
    last_datetime = df.index[-1]
    current_time = datetime.now()
    
    # Calculate time difference
    time_diff = current_time - last_datetime
    
    # If time difference is less than 5 minutes, check if it's incomplete
    if time_diff < timedelta(minutes=5):
        df = df.iloc[:-1]  # Remove last row
        print(f"Removed incomplete data point at {last_datetime}. Last valid data point: {df.index[-1]}")
    else:
        print(f"Data is valid. Last data point: {last_datetime}")
    
    return df

def get_next_m5_datetime(last_datetime):
    """Get the next expected M5 datetime"""
    return last_datetime + timedelta(minutes=5)

def should_update_h1(current_datetime, last_h1_datetime):
    """Check if H1 data needs updating"""
    return current_datetime.hour != last_h1_datetime.hour

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

def get_data_with_retry(symbol, interval, n_bars=1):
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
                logger.info(f"Successfully retrieved data from {exchange}")
                return data
            
        except Exception as e:
            logger.error(f"Error fetching from {exchange}: {str(e)}")
        
        retry_count += 1
        if retry_count < max_retries:
            logger.info(f"Waiting 15 seconds before trying next exchange...")
            time.sleep(15)
    
    return None

def verify_save(filename, expected_data, timeframe):
    """Verify that data was properly saved to CSV and recalculate indicators"""
    try:
        # Read back the saved file
        saved_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        # Verify save
        if saved_data.index[-1] != expected_data.index[-1]:
            logger.error(f"Save verification failed for {filename}")
            return None
            
        # Recalculate indicators on saved data
        logger.info(f"Recalculating indicators for {filename}")
        updated_data = calculate_indicators(saved_data, timeframe)
        if updated_data is None:
            logger.error(f"Failed to recalculate indicators for {filename}")
            return None
            
        # Save with recalculated indicators
        updated_data.to_csv(filename, index=True)
        logger.info(f"Successfully verified and updated indicators for {filename}")
        return updated_data
        
    except Exception as e:
        logger.error(f"Error verifying/updating {filename}: {str(e)}")
        return None

def save_ohlcv_data(filename, new_data, existing_data=None):
    """Save only OHLCV data to CSV"""
    try:
        if existing_data is not None:
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        else:
            combined_data = new_data
            
        # Save only OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        combined_data[ohlcv_columns].to_csv(filename, index=True)
        logger.info(f"Saved OHLCV data to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving OHLCV data to {filename}: {str(e)}")
        return False

def update_indicators(filename, timeframe):
    """Update indicators for entire dataset"""
    try:
        # Read the OHLCV data
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        # Calculate indicators on complete dataset
        updated_data = calculate_indicators(data, timeframe)
        if updated_data is None:
            logger.error(f"Failed to calculate indicators for {filename}")
            return None
            
        # Save complete data with indicators
        updated_data.to_csv(filename, index=True)
        logger.info(f"Updated indicators for {filename}")
        return updated_data
    except Exception as e:
        logger.error(f"Error updating indicators for {filename}: {str(e)}")
        return None

def real_time_monitor():
    """Monitor and update data in real time"""
    global tv
    while True:
        try:
            logger.debug("Starting new monitoring cycle")
            # Read existing data
            m5_data = pd.read_csv('M5.csv', index_col=0, parse_dates=True)
            h1_data = pd.read_csv('H1.csv', index_col=0, parse_dates=True)
            
            last_m5_datetime = m5_data.index[-1]
            last_h1_datetime = h1_data.index[-1]
            next_m5_datetime = get_next_m5_datetime(last_m5_datetime)
            current_time = datetime.now()
            
            logger.debug(f"Last M5: {last_m5_datetime}, Next M5: {next_m5_datetime}, Current: {current_time}")
            
            if current_time >= next_m5_datetime + timedelta(minutes=5):
                logger.info("Time to fetch new data")
                tv = reconnect_tv()
                if tv is None:
                    logger.warning("Reconnection failed, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                try:
                    logger.debug("Fetching new M5 data...")
                    new_m5_data = get_data_with_retry('EURUSD', Interval.in_5_minute)
                    
                    if new_m5_data is not None and not new_m5_data.empty:
                        # First save OHLCV data
                        if save_ohlcv_data('M5.csv', new_m5_data, m5_data):
                            # Then update all indicators
                            updated_m5_data = update_indicators('M5.csv', 'M5')
                            if updated_m5_data is None:
                                logger.error("Failed to update M5 indicators, retrying next cycle")
                                continue
                            
                            m5_data = updated_m5_data
                            logger.info(f"Updated M5 data with new point at {m5_data.index[-1]}")
                            
                            if should_update_h1(m5_data.index[-1], last_h1_datetime):
                                logger.info("Updating H1 data")
                                new_h1_data = get_data_with_retry('EURUSD', Interval.in_1_hour)
                                
                                if new_h1_data is not None and not new_h1_data.empty:
                                    # First save OHLCV data
                                    if save_ohlcv_data('H1.csv', new_h1_data, h1_data):
                                        # Then update all indicators
                                        updated_h1_data = update_indicators('H1.csv', 'H1')
                                        if updated_h1_data is None:
                                            logger.error("Failed to update H1 indicators, retrying next cycle")
                                            continue
                                        
                                        h1_data = updated_h1_data
                                        logger.info(f"Updated H1 data with new point at {h1_data.index[-1]}")
                        else:
                            logger.error("Failed to save M5 OHLCV data, retrying next cycle")
                            continue
                except Exception as e:
                    logger.error(f"Error fetching data: {str(e)}", exc_info=True)
            
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in monitor loop: {str(e)}", exc_info=True)
            time.sleep(60)

# Initialize data files if they don't exist
tv = TvDatafeedLive(username='chiqo-254', password='ChiqoMehum8844')

n_days_back = 10  # specify the number of days back
data1 = get_data_with_retry('EURUSD', Interval.in_5_minute, n_bars=n_days_back * 50)
data2 = get_data_with_retry('EURUSD', Interval.in_1_hour, n_bars=n_days_back * 24)

# Calculate indicators and validate data before saving
data1 = calculate_indicators(data1, 'M5')
data1 = validate_data(data1)
data1.to_csv('M5.csv', index=True)

data2 = calculate_indicators(data2, 'H1')
data2.to_csv('H1.csv', index=True)

# Start real-time monitoring
if __name__ == '__main__':
    import time
    logger.info("Starting trading monitor")
    real_time_monitor()

