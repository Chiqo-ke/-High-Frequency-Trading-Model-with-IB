from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
from talipp.indicators import MACD, RSI, EMA
from talipp.indicator_util import composite_to_lists
import logging
import os
import pandas_ta as ta

logger = logging.getLogger(__name__)
username = 'Chiqo-254'
password = 'ChiqoMehum8844'

tv = TvDatafeed(username,password)

h1_data = tv.get_hist(symbol='EURUSD',exchange='FX_IDC',interval=Interval.in_1_hour,n_bars=5000)
m5_data = tv.get_hist(symbol='EURUSD',exchange='FX_IDC',interval=Interval.in_5_minute,n_bars=8640)

#create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
if h1_data is not None:
    h1_data.to_csv('data/h1_data.csv')
if m5_data is not None: 
    m5_data.to_csv('data/m5_data.csv')
    
    
