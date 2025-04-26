import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_prepare_data(file_path):
    """
    Load the CSV file without headers and apply column mapping
    """
    print(f"Loading data from {file_path}...")
    
    # Define column names since files have no headers
    column_names = ['time', 'o', 'h', 'l', 'c', 'v']
    
    # Load the data without headers
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    # Apply column mapping
    column_mapping = {
        'c': 'close', 
        'o': 'open', 
        'h': 'high', 
        'l': 'low', 
        'v': 'volume'
    }
    
    # Rename columns according to mapping
    df = df.rename(columns=column_mapping)
    
    print(f"Columns in {file_path} after mapping: {df.columns.tolist()}")
    
    # Convert time to datetime format
    try:
        df['time'] = pd.to_datetime(df['time'])
        print("Successfully converted time column to datetime format")
    except Exception as e:
        print(f"Error converting datetime: {e}")
        print("Sample of time values:", df['time'].head())
        raise
    
    # Convert to OANDA time format (typically RFC3339)
    try:
        # OANDA typically uses ISO 8601 / RFC3339 format with 'Z' suffix for UTC
        # Format: YYYY-MM-DDTHH:MM:SS.mmmmmmmZ
        df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        df['time'] = pd.to_datetime(df['time'])  # Convert back to datetime for processing
        print("Successfully converted to OANDA time format")
    except Exception as e:
        print(f"Error formatting to OANDA format: {e}")
        raise
    
    return df

def check_duplicates(df):
    """
    Check for duplicate timestamps in the data
    """
    duplicate_count = df.duplicated(subset=['time']).sum()
    print(f"Found {duplicate_count} duplicate timestamps")
    
    if duplicate_count > 0:
        duplicates = df[df.duplicated(subset=['time'], keep=False)].sort_values(by='time')
        print("\nSample of duplicates:")
        print(duplicates.head())
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=['time'])
        print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
        return df_cleaned
    
    return df

def check_time_consistency(df, timeframe):
    """
    Check for gaps and inconsistencies in the time series
    timeframe should be 'M5' for 5-minute data or 'H1' for hourly data
    """
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # Define expected time delta based on timeframe
    if timeframe == 'M5':
        expected_delta = timedelta(minutes=5)
    elif timeframe == 'H1':
        expected_delta = timedelta(hours=1)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    # Calculate time differences
    df['time_diff'] = df['time'].diff()
    
    # Check for gaps (excluding weekends)
    gaps = []
    inconsistencies = []
    
    for i, row in df.iterrows():
        if i == 0:
            continue
        
        current_time = df.loc[i, 'time']
        prev_time = df.loc[i-1, 'time']
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
    print(f"\nFound {len(gaps)} gaps in the time series")
    if gaps:
        print("Sample of gaps:")
        for i, prev, curr, diff in gaps[:5]:
            print(f"Row {i}: Gap between {prev} and {curr} ({diff})")
    
    print(f"\nFound {len(inconsistencies)} inconsistencies in the time series")
    if inconsistencies:
        print("Sample of inconsistencies:")
        for i, prev, curr, diff in inconsistencies[:5]:
            print(f"Row {i}: Inconsistency between {prev} and {curr} ({diff})")
    
    return df, gaps, inconsistencies

def is_weekend_gap(prev_time, current_time):
    """
    Check if the gap between two timestamps is due to a weekend
    Forex markets typically close on Friday evening and reopen on Sunday evening
    """
    if prev_time.dayofweek == 4 and current_time.dayofweek in [6, 0]:  # Friday to Sunday/Monday
        return True
    return False

def save_cleaned_data(df, original_file):
    """
    Save the cleaned data to a new file
    """
    # Remove temporary columns
    if 'time_diff' in df.columns:
        df = df.drop(columns=['time_diff'])
    
    # Create a new filename
    new_filename = original_file.replace('.csv', '_cleaned.csv')
    
    # Save the data with proper headers
    df.to_csv(new_filename, index=False)
    print(f"\nSaved cleaned data to {new_filename}")
    
    # Generate report
    print(f"\nData Summary:")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Total rows: {len(df)}")
    
    # Check for missing days
    dates = pd.Series(df['time'].dt.date.unique())
    date_range = pd.date_range(start=dates.min(), end=dates.max())
    missing_dates = set(date_range.date) - set(dates)
    
    print(f"Number of unique dates: {len(dates)}")
    missing_weekdays = [d for d in missing_dates if d.weekday() < 5]  # Exclude weekends
    print(f"Missing weekdays: {len(missing_weekdays)}")
    if missing_weekdays and len(missing_weekdays) < 10:
        print("Missing weekdays:", missing_weekdays)

def main():
    # Process M5 data
    print("\n========== Processing 5-minute data ==========")
    m5_file = "M5.csv"
    df_m5 = load_and_prepare_data(m5_file)
    df_m5 = check_duplicates(df_m5)
    df_m5, gaps_m5, inconsistencies_m5 = check_time_consistency(df_m5, 'M5')
    save_cleaned_data(df_m5, m5_file)
    
    # Process H1 data
    print("\n========== Processing 1-hour data ==========")
    h1_file = "H1.csv"
    df_h1 = load_and_prepare_data(h1_file)
    df_h1 = check_duplicates(df_h1)
    df_h1, gaps_h1, inconsistencies_h1 = check_time_consistency(df_h1, 'H1')
    save_cleaned_data(df_h1, h1_file)
    
    print("\n========== Processing Complete ==========")

if __name__ == "__main__":
    main()