import pandas as pd
import numpy as np
import pandas_ta as ta
import pytz
from datetime import datetime, time
import logging
from pathlib import Path
from typing import Dict, List
import os
from config import TECHNICAL_INDICATORS

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_data_dir = "processed_data"
        # Trading sessions in UTC
        self.sessions = {
            'London': {
                'start': time(7, 0),  # 7:00 UTC (8:00 London)
                'end': time(16, 0)    # 16:00 UTC (17:00 London)
            },
            'NewYork': {
                'start': time(13, 0),  # 13:00 UTC (8:00 NY)
                'end': time(22, 0)     # 22:00 UTC (17:00 NY)
            }
        }
        print("\n[DataProcessor] Initialized with trading sessions")
        
    def _is_active_session(self, timestamp) -> dict:
        """Check if timestamp is within trading sessions"""
        current_time = timestamp.time()
        return {
            'London': self.sessions['London']['start'] <= current_time <= self.sessions['London']['end'],
            'NewYork': self.sessions['NewYork']['start'] <= current_time <= self.sessions['NewYork']['end']
        }

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
            
            # Add trading session flags
            print("5. Adding trading session information...")
            df['Time'] = pd.to_datetime(df.index)
            df['London_Session'] = df['Time'].apply(lambda x: self._is_active_session(x)['London'])
            df['NewYork_Session'] = df['Time'].apply(lambda x: self._is_active_session(x)['NewYork'])
            df['Overlap_Session'] = df['London_Session'] & df['NewYork_Session']
            df['Trading_Session'] = df['London_Session'] | df['NewYork_Session']
            
            # Additional session-based features
            df['Hour_of_Day'] = df['Time'].dt.hour
            df['Day_of_Week'] = df['Time'].dt.dayofweek
            
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
