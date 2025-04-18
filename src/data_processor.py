import pandas as pd
import pandas_ta as ta
from typing import Dict, List
import logging
import os
from config import TECHNICAL_INDICATORS

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_data_dir = "processed_data"
        
    def standardize_file(self, filepath: str) -> str:
        """Standardize CSV file with proper headers and return path to processed file"""
        try:
            # Create processed data directory if it doesn't exist
            os.makedirs(self.processed_data_dir, exist_ok=True)  # Fixed parameter name
            
            # Generate processed filepath
            filename = os.path.basename(filepath)
            processed_filepath = os.path.join(self.processed_data_dir, f"processed_{filename}")
            
            # Skip if already processed
            if os.path.exists(processed_filepath):
                return processed_filepath
                
            # Process the file
            expected_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
            df = pd.read_csv(filepath, header=None, names=expected_columns)
            df.to_csv(processed_filepath, index=False)
            
            return processed_filepath
        except Exception as e:
            self.logger.error(f"Error standardizing file {filepath}: {str(e)}")
            raise
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess data with improved error handling"""
        try:
            expected_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
            
            # Update deprecated delim_whitespace to sep
            df = pd.read_csv(filepath, 
                           header=None, 
                           names=expected_columns,
                           sep='\s+',  # Replace delim_whitespace with sep
                           skipinitialspace=True,
                           skip_blank_lines=True)
            
            # Validate data
            if df.empty:
                raise ValueError("Empty dataframe loaded")
            
            # Clean and parse datetime - handle malformed data
            df['Time'] = pd.to_datetime(df['Time'], format='mixed')
            
            # Clean numeric columns - remove any remaining whitespace and convert to numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            
            # Remove any rows with NaN values after conversion
            df = df.dropna()
            
            # Ensure Time column is unique before technical analysis
            df = df.drop_duplicates(subset=['Time'], keep='first')
            df.set_index('Time', inplace=True)
            df.sort_index(inplace=True)
            
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
            # RSI
            df['RSI'] = ta.rsi(df['Close'], **TECHNICAL_INDICATORS['RSI'])
            
            # MACD
            macd = ta.macd(df['Close'], **TECHNICAL_INDICATORS['MACD'])
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands
            bb = ta.bbands(df['Close'], **TECHNICAL_INDICATORS['BB'])
            df = pd.concat([df, bb], axis=1)
            
            # ATR
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 
                             **TECHNICAL_INDICATORS['ATR'])
            return df

        except Exception as e:
            self.logger.error(f"Error in technical analysis: {str(e)}")
            return df.copy()

    def get_trading_session(self, time) -> str:
        """Determine trading session based on time"""
        hour = time.hour
        if 0 <= hour < 9:
            return 'ASIAN'
        elif 8 <= hour < 17:
            return 'EUROPEAN'
        elif 13 <= hour < 22:
            return 'AMERICAN'
        return 'CLOSED'
