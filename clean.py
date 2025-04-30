import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forex_data_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Create data directory if it doesn't exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def store_dataframe_as_csv(key_prefix, df):
    """
    Store a DataFrame as CSV with a key prefix
    """
    try:
        # Create file path
        file_path = os.path.join(DATA_DIR, f"{key_prefix}.csv")
        
        # Store the DataFrame
        df.to_csv(file_path, index=True)
        logger.info(f"Successfully stored DataFrame to: {file_path}")
        
        # Store metadata about the DataFrame
        metadata = {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metadata as CSV
        metadata_df = pd.DataFrame([metadata])
        metadata_path = os.path.join(DATA_DIR, f"{key_prefix}_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        
        return True
    except Exception as e:
        logger.error(f"Error storing DataFrame as CSV: {e}")
        return False

def load_dataframe_from_csv(key_prefix):
    """
    Load a DataFrame from CSV using the key prefix
    """
    try:
        # Create file path
        file_path = os.path.join(DATA_DIR, f"{key_prefix}.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"No CSV file found at: {file_path}")
            return None
            
        # Load the DataFrame
        df = pd.read_csv(file_path, parse_dates=True)
        
        # Check if there's an index column from to_csv
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'time'})
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        logger.info(f"Successfully loaded DataFrame from: {file_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from CSV: {e}")
        return None

def update_dataframe_in_csv(key_prefix, new_data):
    """
    Update an existing DataFrame in CSV with new data
    """
    # Load existing data
    existing_df = load_dataframe_from_csv(key_prefix)
    
    if existing_df is None:
        # If no existing data, just store the new data
        store_dataframe_as_csv(key_prefix, new_data)
        return new_data
    
    # Ensure the new_data is a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        else:
            logger.error("New data must be a DataFrame or dict")
            return existing_df
    
    # Get time column name
    time_col = existing_df.index.name
    
    # Reset index to make time a column for concatenation
    existing_df = existing_df.reset_index()
    if hasattr(new_data, 'index') and new_data.index.name == time_col:
        new_data = new_data.reset_index()
    
    # Ensure time column is in datetime format
    if time_col in existing_df.columns and time_col in new_data.columns:
        existing_df[time_col] = pd.to_datetime(existing_df[time_col])
        new_data[time_col] = pd.to_datetime(new_data[time_col])
    
    # Combine and deduplicate
    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    if time_col in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=[time_col])
        combined_df = combined_df.sort_values(by=time_col).reset_index(drop=True)
        combined_df.set_index(time_col, inplace=True)
    
    # Store updated DataFrame
    store_dataframe_as_csv(key_prefix, combined_df)
    
    logger.info(f"Updated DataFrame in CSV with {len(new_data)} new rows")
    return combined_df

def load_and_prepare_data(file_path):
    """
    Load the CSV file and apply column mapping
    """
    logger.info(f"Loading data from {file_path}...")
    
    # Read CSV with headers
    df = pd.read_csv(file_path)
    
    # Drop the symbol column if it exists
    if 'symbol' in df.columns:
        df = df.drop('symbol', axis=1)
        logger.info("Dropped symbol column")
    
    # Apply column mapping for standardization
    column_mapping = {
        'datetime': 'time',
        'close': 'close', 
        'open': 'open', 
        'high': 'high', 
        'low': 'low', 
        'volume': 'volume',
        'c': 'close', 
        'o': 'open', 
        'h': 'high', 
        'l': 'low', 
        'v': 'volume'
    }
    
    # Rename columns according to mapping (only for columns that exist)
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    logger.info(f"Columns after mapping: {df.columns.tolist()}")
    
    # Convert time to datetime format
    try:
        # Parse datetime directly since it's in a standard format
        datetime_col = 'time' if 'time' in df.columns else 'datetime'
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        logger.info("Successfully parsed dates")
        
        # Convert to OANDA time format
        df[datetime_col] = df[datetime_col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        logger.info("Successfully converted to OANDA time format")
        
    except Exception as e:
        logger.error(f"Error processing datetime: {e}")
        logger.error(f"Sample of time values: {df[datetime_col].head()}")
        raise
    
    return df

def process_new_order_data(new_order_data, timeframe):
    """
    Process new order data as it comes in and update the CSV store
    
    Parameters:
        new_order_data: Dictionary or DataFrame containing new order data
        timeframe: 'M5' for 5-minute or 'H1' for hourly data
    """
    logger.info(f"Processing new {timeframe} order data...")
    
    # Convert dict to DataFrame if needed
    if isinstance(new_order_data, dict):
        new_df = pd.DataFrame([new_order_data])
    else:
        new_df = new_order_data.copy()
    
    # Get key for CSV based on timeframe
    csv_key = f"forex_data_{timeframe}"
    
    # Update the dataframe in CSV
    updated_df = update_dataframe_in_csv(csv_key, new_df)
    
    # Check for duplicates and time consistency
    updated_df = check_duplicates(updated_df)
    updated_df, _, _ = check_time_consistency(updated_df, timeframe)
    
    # Store the cleaned data back to CSV
    store_dataframe_as_csv(f"forex_data_{timeframe}_cleaned", updated_df)
    
    logger.info(f"Processed and stored new {timeframe} data in CSV")
    
    return updated_df

def check_duplicates(df):
    """
    Check for duplicate timestamps in the data
    """
    # Reset index if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        time_col = df.index.name or 'time'
        df = df.reset_index()
    else:
        time_col = 'time' if 'time' in df.columns else 'datetime'
    
    duplicate_count = df.duplicated(subset=[time_col]).sum()
    logger.info(f"Found {duplicate_count} duplicate timestamps")
    
    if duplicate_count > 0:
        duplicates = df[df.duplicated(subset=[time_col], keep=False)].sort_values(by=time_col)
        logger.info(f"\nSample of duplicates:\n{duplicates.head()}")
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=[time_col])
        logger.info(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
        
        # Set index back to time column
        df_cleaned.set_index(time_col, inplace=True)
        return df_cleaned
    
    # Set index back to time column if it wasn't already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index(time_col, inplace=True)
    
    return df

def is_weekend_gap(prev_time, current_time):
    """
    Check if the gap between two timestamps is due to a weekend
    Forex markets typically close on Friday evening and reopen on Sunday evening
    """
    if prev_time.dayofweek == 4 and current_time.dayofweek in [6, 0]:  # Friday to Sunday/Monday
        return True
    return False

def check_time_consistency(df, timeframe):
    """
    Check for gaps and inconsistencies in the time series
    timeframe should be 'M5' for 5-minute data or 'H1' for hourly data
    """
    # Make sure we're working with the index as time
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = 'time' if 'time' in df.columns else 'datetime'
        df = df.sort_values(by=time_col).set_index(time_col)
    else:
        df = df.sort_index()
    
    # Define expected time delta based on timeframe
    if timeframe == 'M5':
        expected_delta = timedelta(minutes=5)
    elif timeframe == 'H1':
        expected_delta = timedelta(hours=1)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    # Calculate time differences
    time_diff = df.index.to_series().diff()
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df['time_diff'] = time_diff
    
    # Check for gaps (excluding weekends)
    gaps = []
    inconsistencies = []
    
    for i in range(1, len(df)):
        current_time = df.index[i]
        prev_time = df.index[i-1]
        diff = current_time - prev_time
        
        # Skip weekend checks
        if is_weekend_gap(prev_time, current_time):
            continue
        
        # Check if the difference is not the expected delta
        if diff != expected_delta:
            # If it's a multiple of the expected delta, it's a gap
            if diff > expected_delta and diff % expected_delta == timedelta(0):
                gaps.append((i, prev_time, current_time, diff))
            else:
                inconsistencies.append((i, prev_time, current_time, diff))
    
    # Report findings
    logger.info(f"\nFound {len(gaps)} gaps in the time series")
    if gaps:
        logger.info("Sample of gaps:")
        for i, prev, curr, diff in gaps[:5]:
            logger.info(f"Row {i}: Gap between {prev} and {curr} ({diff})")
    
    logger.info(f"\nFound {len(inconsistencies)} inconsistencies in the time series")
    if inconsistencies:
        logger.info("Sample of inconsistencies:")
        for i, prev, curr, diff in inconsistencies[:5]:
            logger.info(f"Row {i}: Inconsistency between {prev} and {curr} ({diff})")
    
    return df, gaps, inconsistencies

def calculate_indicators():
    """
    Calculate technical indicators for both timeframes using data from CSV
    """
    logger.info("\n========== Calculating Technical Indicators ==========")
    
    # Load the cleaned data from CSV
    h1_data = load_dataframe_from_csv("forex_data_H1_cleaned")
    m5_data = load_dataframe_from_csv("forex_data_M5_cleaned")
    
    if h1_data is None or m5_data is None:
        logger.error("Could not load data from CSV for indicator calculation")
        return None, None
    
    # Ensure we're working with DatetimeIndex
    if not isinstance(h1_data.index, pd.DatetimeIndex):
        datetime_col = 'time' if 'time' in h1_data.columns else 'datetime'
        h1_data[datetime_col] = pd.to_datetime(h1_data[datetime_col])
        h1_data.set_index(datetime_col, inplace=True)
    
    if not isinstance(m5_data.index, pd.DatetimeIndex):
        datetime_col = 'time' if 'time' in m5_data.columns else 'datetime'
        m5_data[datetime_col] = pd.to_datetime(m5_data[datetime_col])
        m5_data.set_index(datetime_col, inplace=True)
    
    logger.info("Calculating 1H indicators...")
    # Trend Bias: 8-period EMA on 1-hour chart
    h1_data['ema_8'] = ta.ema(h1_data['close'], length=8)
    h1_data['trend_bias'] = np.where(h1_data['close'] > h1_data['ema_8'], 'bullish', 'bearish')
    
    logger.info("Calculating 5M indicators...")
    # Entry Points: 5-period EMA on 5-minute chart
    m5_data['ema_5'] = ta.ema(m5_data['close'], length=5)
    m5_data['price_crossed_above_ema'] = np.where(
        (m5_data['close'] > m5_data['ema_5']) & (m5_data['close'].shift(1) <= m5_data['ema_5'].shift(1)), 
        True, False
    )
    m5_data['price_crossed_below_ema'] = np.where(
        (m5_data['close'] < m5_data['ema_5']) & (m5_data['close'].shift(1) >= m5_data['ema_5'].shift(1)), 
        True, False
    )
    
    # RSI indicator for divergence (alternative entry)
    m5_data['rsi_9'] = ta.rsi(m5_data['close'], length=9)
    
    # Exit trades: MACD (8, 17, 9) on 5-minute chart
    macd = ta.macd(m5_data['close'], fast=8, slow=17, signal=9)
    m5_data = m5_data.join(macd)
    
    # Identify MACD crossovers
    m5_data['macd_cross_below'] = np.where(
        (m5_data['MACD_8_17_9'] < m5_data['MACDs_8_17_9']) & 
        (m5_data['MACD_8_17_9'].shift(1) >= m5_data['MACDs_8_17_9'].shift(1)), 
        True, False
    )
    m5_data['macd_cross_above'] = np.where(
        (m5_data['MACD_8_17_9'] > m5_data['MACDs_8_17_9']) & 
        (m5_data['MACD_8_17_9'].shift(1) <= m5_data['MACDs_8_17_9'].shift(1)), 
        True, False
    )  # Fixed syntax error in original code
    
    # Store results with indicators in CSVs
    store_dataframe_as_csv("forex_data_H1_indicators", h1_data)
    store_dataframe_as_csv("forex_data_M5_indicators", m5_data)
    
    logger.info("1H Data Sample:")
    logger.info(h1_data[['close', 'ema_8', 'trend_bias']].tail())
    
    logger.info("\n5M Data Sample:")
    logger.info(m5_data[['close', 'ema_5', 'rsi_9', 'MACD_8_17_9', 'MACDs_8_17_9', 
                      'macd_cross_above', 'macd_cross_below']].tail())
    
    logger.info("Indicators calculated and stored in CSVs!")
    
    return h1_data, m5_data

def create_signals_dataframe():
    """
    Create a signals dataframe that combines indicators from both timeframes
    while preventing forward-looking bias
    """
    logger.info("\n========== Creating Signals DataFrame ==========")
    logger.info("Ensuring no forward-looking bias in signal generation...")
    
    # Load data with indicators from CSVs
    h1_data = load_dataframe_from_csv("forex_data_H1_indicators")
    m5_data = load_dataframe_from_csv("forex_data_M5_indicators")
    
    if h1_data is None or m5_data is None:
        logger.error("Could not load indicator data from CSVs")
        return None
    
    # Create signals DataFrame with data that would have been available at each point
    signals_df = pd.DataFrame({
        'datetime': m5_data.index,
        'close': m5_data['close'],
        'ema_5': m5_data['ema_5'].shift(1),  # Use previous period's EMA
        'rsi_9': m5_data['rsi_9'].shift(1),  # Use previous period's RSI
        'macd': m5_data['MACD_8_17_9'].shift(1),  # Use previous period's MACD
        'macd_signal': m5_data['MACDs_8_17_9'].shift(1),  # Use previous period's MACD signal
        'price_crossed_above_ema': m5_data['price_crossed_above_ema'].shift(1),
        'price_crossed_below_ema': m5_data['price_crossed_below_ema'].shift(1),
        'macd_cross_above': m5_data['macd_cross_above'].shift(1),
        'macd_cross_below': m5_data['macd_cross_below'].shift(1)
    })
    
    # Add hourly trend bias by resampling to 5-minute timeframe
    # Use previous hour's bias to prevent looking ahead
    hourly_bias = h1_data['trend_bias'].shift(1).reindex(m5_data.index, method='ffill')
    signals_df['trend_bias'] = hourly_bias
    
    # Drop the first row since it will contain NaN values due to shifting
    signals_df = signals_df.dropna()
    
    # Add warning in logs about preventing forward-looking bias
    logger.info("Forward-looking bias prevention measures applied:")
    logger.info("1. All indicator values shifted by 1 period")
    logger.info("2. Using previous hour's trend bias")
    logger.info("3. First row dropped due to NaN values from shifting")
    
    # Store signals in CSV
    store_dataframe_as_csv("forex_data_trading_signals", signals_df)
    logger.info("Bias-free signals DataFrame created and stored in CSV!")
    
    return signals_df

def visualize_indicators(h1_sample, m5_sample, period=50, save_path='indicator_analysis.png'):
    """
    Visualize the indicators for analysis
    """
    logger.info("\n========== Visualizing Indicators ==========")
    
    # 1H Chart with EMA
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.title('1H Chart - Trend Bias')
    plt.plot(h1_sample.index[-period:], h1_sample['close'][-period:], label='Close')
    plt.plot(h1_sample.index[-period:], h1_sample['ema_8'][-period:], label='EMA(8)')
    plt.legend()
    plt.grid(True)
    
    # 5M Chart with Entry/Exit signals
    plt.subplot(2, 1, 2)
    plt.title('5M Chart - Entry & Exit Points')
    plt.plot(m5_sample.index[-period*12:], m5_sample['close'][-period*12:], label='Close')
    plt.plot(m5_sample.index[-period*12:], m5_sample['ema_5'][-period*12:], label='EMA(5)')
    
    # Plot entry signals
    entries_long = m5_sample[-period*12:][m5_sample['price_crossed_above_ema'][-period*12:]]
    entries_short = m5_sample[-period*12:][m5_sample['price_crossed_below_ema'][-period*12:]]
    plt.scatter(entries_long.index, entries_long['close'], color='green', marker='^', s=100, label='Long Entry')
    plt.scatter(entries_short.index, entries_short['close'], color='red', marker='v', s=100, label='Short Entry')
    
    # Plot exit signals
    exits_long = m5_sample[-period*12:][m5_sample['macd_cross_below'][-period*12:]]
    exits_short = m5_sample[-period*12:][m5_sample['macd_cross_above'][-period*12:]]
    plt.scatter(exits_long.index, exits_long['close'], color='orange', marker='x', s=100, label='Long Exit')
    plt.scatter(exits_short.index, exits_short['close'], color='purple', marker='x', s=100, label='Short Exit')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Indicator visualization saved to {save_path}")
    
    # Close the plot to free memory
    plt.close()

def initial_data_load():
    """
    Initial load of data from CSV files to data directory
    """
    logger.info("\n========== Initial Data Load to CSV Files ==========")
    
    # Process M5 data
    logger.info("\n========== Processing 5-minute data ==========")
    m5_file = "M5.csv"  # Original input file
    df_m5 = load_and_prepare_data(m5_file)
    df_m5 = check_duplicates(df_m5)
    df_m5, gaps_m5, inconsistencies_m5 = check_time_consistency(df_m5, 'M5')
    
    # Store M5 data as CSV
    store_dataframe_as_csv("forex_data_M5_cleaned", df_m5)
    
    # Process H1 data
    logger.info("\n========== Processing 1-hour data ==========")
    h1_file = "H1.csv"  # Original input file
    df_h1 = load_and_prepare_data(h1_file)
    df_h1 = check_duplicates(df_h1)
    df_h1, gaps_h1, inconsistencies_h1 = check_time_consistency(df_h1, 'H1')
    
    # Store H1 data as CSV
    store_dataframe_as_csv("forex_data_H1_cleaned", df_h1)
    
    logger.info("Initial data loaded into CSV files successfully")
    
    return df_h1, df_m5

def process_new_order(order_data, timeframe):
    """
    Process a new order and update indicators and signals
    
    Parameters:
        order_data: Dictionary with order data
        timeframe: 'M5' or 'H1'
    """
    logger.info(f"Processing new order for {timeframe} timeframe")
    
    # Update data with new order
    process_new_order_data(order_data, timeframe)
    
    # Recalculate indicators
    h1_data, m5_data = calculate_indicators()
    
    # Update signals
    signals_df = create_signals_dataframe()
    
    logger.info(f"Updated indicators and signals with new {timeframe} order")
    
    return signals_df

def main():
    """
    Main function to run the entire forex data processing pipeline with CSV storage
    """
    try:
        logger.info("==================================================")
        logger.info("Starting Forex Data Processing Pipeline with CSV Storage")
        logger.info("==================================================")
        
        # Initial load from CSV files to data directory
        h1_data, m5_data = initial_data_load()
        
        # Calculate indicators
        h1_data, m5_data = calculate_indicators()
        
        # Create signals dataframe
        signals_df = create_signals_dataframe()
        
        # Optional: Visualize indicators
        visualize_indicators(h1_data, m5_data)
        
        logger.info("\n========== Processing Complete ==========")
        logger.info(f"Data stored in CSV files in '{DATA_DIR}' directory:")
        logger.info(f"1. forex_data_H1_cleaned.csv - Cleaned hourly data")
        logger.info(f"2. forex_data_M5_cleaned.csv - Cleaned 5-minute data")
        logger.info(f"3. forex_data_H1_indicators.csv - Hourly data with indicators")
        logger.info(f"4. forex_data_M5_indicators.csv - 5-minute data with indicators")
        logger.info(f"5. forex_data_trading_signals.csv - Combined trading signals")
        logger.info(f"6. indicator_analysis.png - Visual chart")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()