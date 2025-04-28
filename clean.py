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

def check_duplicates(df):
    """
    Check for duplicate timestamps in the data
    """
    time_col = 'time' if 'time' in df.columns else 'datetime'
    duplicate_count = df.duplicated(subset=[time_col]).sum()
    logger.info(f"Found {duplicate_count} duplicate timestamps")
    
    if duplicate_count > 0:
        duplicates = df[df.duplicated(subset=[time_col], keep=False)].sort_values(by=time_col)
        logger.info(f"\nSample of duplicates:\n{duplicates.head()}")
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=[time_col])
        logger.info(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
        return df_cleaned
    
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
    time_col = 'time' if 'time' in df.columns else 'datetime'
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Define expected time delta based on timeframe
    if timeframe == 'M5':
        expected_delta = timedelta(minutes=5)
    elif timeframe == 'H1':
        expected_delta = timedelta(hours=1)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    # Calculate time differences
    df['time_diff'] = df[time_col].diff()
    
    # Check for gaps (excluding weekends)
    gaps = []
    inconsistencies = []
    
    for i, row in df.iterrows():
        if i == 0:
            continue
        
        current_time = df.loc[i, time_col]
        prev_time = df.loc[i-1, time_col]
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

def save_cleaned_data(df, original_file):
    """
    Save the cleaned data to a new file
    """
    # Remove temporary columns
    if 'time_diff' in df.columns:
        df = df.drop(columns=['time_diff'])
    
    # Check if dataframe is empty
    if len(df) == 0:
        logger.warning(f"\nWarning: No valid data to save for {original_file}")
        return
    
    # Create a new filename
    new_filename = original_file.replace('.csv', '_cleaned.csv')
    
    # Save the data with proper headers
    df.to_csv(new_filename, index=False)
    logger.info(f"\nSaved cleaned data to {new_filename}")
    
    # Generate report
    time_col = 'time' if 'time' in df.columns else 'datetime'
    logger.info(f"\nData Summary:")
    logger.info(f"Time range: {df[time_col].min()} to {df[time_col].max()}")
    logger.info(f"Total rows: {len(df)}")
    
    try:
        # Check for missing days
        dates = pd.Series(df[time_col].dt.date.unique())
        date_range = pd.date_range(start=dates.min(), end=dates.max())
        missing_dates = set(date_range.date) - set(dates)
        
        logger.info(f"Number of unique dates: {len(dates)}")
        missing_weekdays = [d for d in missing_dates if d.weekday() < 5]  # Exclude weekends
        logger.info(f"Missing weekdays: {len(missing_weekdays)}")
        if missing_weekdays and len(missing_weekdays) < 10:
            logger.info(f"Missing weekdays: {missing_weekdays}")
    except Exception as e:
        logger.warning(f"Could not analyze missing dates: {e}")
    
    return new_filename

def calculate_indicators(h1_file, m5_file):
    """
    Calculate technical indicators for both timeframes
    """
    logger.info("\n========== Calculating Technical Indicators ==========")
    
    # Load the cleaned data
    h1_data = pd.read_csv(h1_file)
    m5_data = pd.read_csv(m5_file)
    
    # Convert datetime columns
    datetime_col_h1 = 'time' if 'time' in h1_data.columns else 'datetime'
    datetime_col_m5 = 'time' if 'time' in m5_data.columns else 'datetime'
    
    h1_data[datetime_col_h1] = pd.to_datetime(h1_data[datetime_col_h1])
    h1_data.set_index(datetime_col_h1, inplace=True)
    
    m5_data[datetime_col_m5] = pd.to_datetime(m5_data[datetime_col_m5])
    m5_data.set_index(datetime_col_m5, inplace=True)
    
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
    )
    
    # Save results with indicators
    h1_output = h1_file.replace('_cleaned.csv', '_with_indicators.csv')
    m5_output = m5_file.replace('_cleaned.csv', '_with_indicators.csv')
    
    h1_data.to_csv(h1_output)
    m5_data.to_csv(m5_output)
    
    logger.info("1H Data Sample:")
    logger.info(h1_data[['close', 'ema_8', 'trend_bias']].tail())
    
    logger.info("\n5M Data Sample:")
    logger.info(m5_data[['close', 'ema_5', 'rsi_9', 'MACD_8_17_9', 'MACDs_8_17_9', 
                      'macd_cross_above', 'macd_cross_below']].tail())
    
    logger.info("Indicators calculated and saved to CSV files!")
    
    return h1_data, m5_data

def create_signals_dataframe(h1_data, m5_data):
    """
    Create a signals dataframe that combines indicators from both timeframes
    while preventing forward-looking bias
    """
    logger.info("\n========== Creating Signals DataFrame ==========")
    logger.info("Ensuring no forward-looking bias in signal generation...")
    
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
    
    signals_df.to_csv('trading_signals.csv')
    logger.info("Bias-free signals DataFrame created and saved!")
    
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

def main():
    """
    Main function to run the entire forex data processing pipeline
    """
    try:
        logger.info("==================================================")
        logger.info("Starting Forex Data Processing Pipeline")
        logger.info("==================================================")
        
        # Process M5 data
        logger.info("\n========== Processing 5-minute data ==========")
        m5_file = "M5.csv"
        df_m5 = load_and_prepare_data(m5_file)
        df_m5 = check_duplicates(df_m5)
        df_m5, gaps_m5, inconsistencies_m5 = check_time_consistency(df_m5, 'M5')
        m5_cleaned_file = save_cleaned_data(df_m5, m5_file)
        
        # Process H1 data
        logger.info("\n========== Processing 1-hour data ==========")
        h1_file = "H1.csv"
        df_h1 = load_and_prepare_data(h1_file)
        df_h1 = check_duplicates(df_h1)
        df_h1, gaps_h1, inconsistencies_h1 = check_time_consistency(df_h1, 'H1')
        h1_cleaned_file = save_cleaned_data(df_h1, h1_file)
        
        # Calculate indicators
        h1_data, m5_data = calculate_indicators(h1_cleaned_file, m5_cleaned_file)
        
        # Create signals dataframe
        signals_df = create_signals_dataframe(h1_data, m5_data)
        
        # Optional: Visualize indicators
        visualize_indicators(h1_data, m5_data)
        
        logger.info("\n========== Processing Complete ==========")
        logger.info(f"Output files created:")
        logger.info(f"1. {h1_cleaned_file}")
        logger.info(f"2. {m5_cleaned_file}")
        logger.info(f"3. {h1_cleaned_file.replace('_cleaned.csv', '_with_indicators.csv')}")
        logger.info(f"4. {m5_cleaned_file.replace('_cleaned.csv', '_with_indicators.csv')}")
        logger.info(f"5. trading_signals.csv")
        logger.info(f"6. indicator_analysis.png")
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()