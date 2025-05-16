import pandas as pd
import numpy as np
from talipp.indicators import MACD, RSI, EMA
from talipp.indicator_util import composite_to_lists
import logging

logger = logging.getLogger(__name__)

def calculate_indicators(df, timeframe='H1'):
    """Calculate indicators if data exists and has enough points"""
    if df is None or df.empty:
        logger.error("No data available for indicator calculation")
        return None
        
    close_values = df['close'].tolist()
    if len(close_values) < 17:
        logger.error(f"Not enough data points for indicators")
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
        
        # Calculate crossovers
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

def update_indicators(filename, timeframe):
    """Update indicators for entire dataset"""
    try:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        updated_data = calculate_indicators(data, timeframe)
        if updated_data is None:
            return None
            
        updated_data.to_csv(filename, index=True)
        logger.info(f"Updated indicators for {filename}")
        logger.debug(f"Last 3 rows with indicators:\n{updated_data.tail(3)}")
        return updated_data
    except Exception as e:
        logger.error(f"Error updating indicators for {filename}: {str(e)}")
        return None

class StateHistory:
    """Track previous states for crossover calculations"""
    def __init__(self):
        self.prev_close = None
        self.prev_ema5 = None
        self.prev_macd_line = None
        self.prev_macd_signal = None
        
    def update(self, close, ema5, macd_line, macd_signal):
        self.prev_close = close
        self.prev_ema5 = ema5
        self.prev_macd_line = macd_line
        self.prev_macd_signal = macd_signal

def rule_based_policy(state, state_history):
    """
    Maps market states to trading actions
    
    Args:
        state: tuple(h1_trend_bias, m5_close, m5_ema5, m5_rsi_9, m5_macd_line, m5_macd_signal, current_position)
        state_history: StateHistory object tracking previous values
        
    Returns:
        action (int): 0=Flat, 1=Long, 2=Short
    """
    try:
        h1_trend_bias, m5_close, m5_ema5, m5_rsi_9, m5_macd_line, m5_macd_signal, current_position = state
        
        # Initialize crossover flags
        price_crossed_above_ema = False
        price_crossed_below_ema = False
        macd_cross_below = False
        macd_cross_above = False
        
        # Calculate crossovers if we have previous state
        if state_history.prev_close is not None:
            price_crossed_above_ema = (m5_close > m5_ema5 and 
                                     state_history.prev_close <= state_history.prev_ema5)
            price_crossed_below_ema = (m5_close < m5_ema5 and 
                                     state_history.prev_close >= state_history.prev_ema5)
            macd_cross_below = (m5_macd_line < m5_macd_signal and 
                              state_history.prev_macd_line >= state_history.prev_macd_signal)
            macd_cross_above = (m5_macd_line > m5_macd_signal and 
                              state_history.prev_macd_line <= state_history.prev_macd_signal)
        
        # Update state history
        state_history.update(m5_close, m5_ema5, m5_macd_line, m5_macd_signal)
        
        # Trading logic
        if current_position == 0:  # No position
            if h1_trend_bias == 1 and price_crossed_above_ema and m5_rsi_9 > 30:
                return 1  # Long
            elif h1_trend_bias == 0 and price_crossed_below_ema and m5_rsi_9 < 70:
                return 2  # Short
        elif current_position == 1:  # Long position
            if macd_cross_below:
                return 0  # Close (go flat)
            return 1  # Keep long
        elif current_position == 2:  # Short position
            if macd_cross_above:
                return 0  # Close (go flat)
            return 2  # Keep short
            
        return 0  # Default to flat
        
    except Exception as e:
        logger.error(f"Error in rule_based_policy: {str(e)}")
        return 0  # Default to flat on error
