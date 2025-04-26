import pandas as pd

def check_duplicates(file_path, timeframe):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Find duplicate timestamps
    duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
    
    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} duplicate entries in {timeframe} timeframe:")
        print("=" * 80)
        
        # Group duplicates by datetime
        grouped = duplicates.groupby('datetime')
        
        for datetime, group in grouped:
            # Get the line numbers (adding 2 because of 0-based index and header row)
            line_numbers = [str(idx + 2) for idx in df[df['datetime'] == datetime].index]
            
            print(f"Duplicate entries found at lines: {', '.join(line_numbers)}")
            print(f"Timestamp: {datetime}")
            print(f"Open: {group['open'].iloc[0]:.5f}")
            print(f"High: {group['high'].iloc[0]:.5f}")
            print(f"Low: {group['low'].iloc[0]:.5f}")
            print(f"Close: {group['close'].iloc[0]:.5f}")
            print(f"Volume: {group['volume'].iloc[0]}")
            print("-" * 40)
    else:
        print(f"\nNo duplicates found in {timeframe} timeframe")

def main():
    # Check both files
    print("Checking for duplicate timestamps in the data files...")
    
    try:
        check_duplicates('H1.csv', 'H1')
        check_duplicates('M5.csv', 'M5')
        
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSV files. Make sure both H1.csv and M5.csv exist in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
