import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_dataframe_processing(df, timeframe):
    """
    Debug function to print detailed information about dataframe at each processing step
    """
    logger.info(f"\nDebugging {timeframe} timeframe processing:")
    
    # Check original data
    logger.info(f"Original data sample:\n{df.head()}")
    logger.info(f"Time column dtype: {df['Time'].dtype}")
    logger.info(f"Data shape: {df.shape}")
    
    # Convert time if it's not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        try:
            df['Time'] = pd.to_datetime(df['Time'])
            logger.info("Time conversion successful")
            logger.info(f"First timestamp: {df['Time'].iloc[0]}")
            logger.info(f"Last timestamp: {df['Time'].iloc[-1]}")
        except Exception as e:
            logger.error(f"Time conversion failed: {e}")
            return
    
    # Resample check
    try:
        resampled = df.resample(timeframe, on='Time').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Spread': 'mean'
        })
        logger.info(f"Resampled shape: {resampled.shape}")
        logger.info(f"Resampled sample:\n{resampled.head()}")
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return

    # Calculate RSI (if this is part of your processing)
    try:
        diff = resampled['Close'].diff()
        gain = diff.where(diff > 0, 0)
        loss = -diff.where(diff < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        logger.info(f"RSI calculation successful")
        logger.info(f"RSI sample:\n{rsi.head()}")
        logger.info(f"Number of valid RSI values: {rsi.notna().sum()}")
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")

if __name__ == "__main__":
    try:
        data_dir = "data"
        timeframe_files = {
            '5min': 'M5.csv',
            '15min': 'M15.csv',
            '30min': 'M30.csv',
            'h': 'H1.csv',
            '4h': 'H4.csv'
        }
        
        # Define expected columns
        expected_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
        
        for tf, filename in timeframe_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"\nProcessing {filename}")
                # Read tab-delimited file without headers
                df = pd.read_csv(filepath, 
                               sep='\t',
                               header=None, 
                               names=expected_columns,
                               engine='python',
                               skipinitialspace=True)
                debug_dataframe_processing(df.copy(), tf)
            else:
                logger.warning(f"File not found: {filepath}")
                
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        logger.error(f"Error details: {str(e)}")
        raise
