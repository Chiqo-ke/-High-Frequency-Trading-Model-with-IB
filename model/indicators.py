import os
import numpy as np
import pandas_ta as ta
import pandas as pd

# Load the data
h1_data = pd.read_csv('data/h1_data.csv')
m5_data = pd.read_csv('data/m5_data.csv')

# Calculate H1 trend bias (simplified example - you may need to customize this)
def calculate_h1_trend_bias(h1_data):
    # Trend Bias: 8-period EMA on 1-hour chart
    h1_data['ema_8'] = ta.ema(h1_data['close'], length=8)
    h1_data['trend_bias'] = np.where(
        h1_data['close'] > h1_data['ema_8'], 
        'bullish', 
        'bearish'
    )
    return h1_data['trend_bias'].iloc[-1]

# Calculate technical indicators
def calculate_and_save_indicators(file_path, is_m5=False, h1_trend_bias=None):
    try:
        df = pd.read_csv(file_path)
        if 'close' not in df.columns:
            raise ValueError(f"'close' column not found in {file_path}")
        
        if is_m5:
            print("Calculating 5M indicators...")
            # Add H1 trend bias to M5 data if applicable
            if h1_trend_bias is not None:
                df['trend_bias'] = h1_trend_bias
        else:
            print("Calculating 1H indicators...")
            
        # EMA
        df['ema_5'] = ta.ema(df['close'], length=5)
        
        # EMA crossover signals
        df['price_crossed_above_ema'] = np.where(
            (df['close'] > df['ema_5']) & (df['close'].shift(1) <= df['ema_5'].shift(1)),
            True,
            False
        )
        df['price_crossed_below_ema'] = np.where(
            (df['close'] < df['ema_5']) & (df['close'].shift(1) >= df['ema_5'].shift(1)),
            True,
            False
        )
        
        # RSI indicator for divergence
        df['rsi_9'] = ta.rsi(df['close'], length=9)
        
        # MACD for exit trades
        macd = ta.macd(df['close'], fast=8, slow=17, signal=9)
        df = df.join(macd)
        
        # Identify MACD crossovers
        df['macd_cross_below'] = np.where(
            (df['MACD_8_17_9'] < df['MACDs_8_17_9']) & 
            (df['MACD_8_17_9'].shift(1) >= df['MACDs_8_17_9'].shift(1)),
            True,
            False
        )
        df['macd_cross_above'] = np.where(
            (df['MACD_8_17_9'] > df['MACDs_8_17_9']) & 
            (df['MACD_8_17_9'].shift(1) <= df['MACDs_8_17_9'].shift(1)),
            True,
            False
        )
        
        # Save back to the same file
        df.to_csv(file_path, index=False)
        print(f"Indicators calculated and saved for {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# First calculate H1 data and get trend bias
calculate_and_save_indicators('data/h1_data.csv')
h1_trend_bias = calculate_h1_trend_bias(pd.read_csv('data/h1_data.csv'))

# Then calculate M5 data with H1 trend bias
calculate_and_save_indicators('data/m5_data.csv', is_m5=True, h1_trend_bias=h1_trend_bias)
