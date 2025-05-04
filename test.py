from ib_insync import *
from datetime import datetime, timedelta
import pytz

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

# Get current time in UTC
now = datetime.now(pytz.UTC)
# Subtract 5 minutes to ensure we get the last completed candle
end_time = now - timedelta(minutes=5)

contract = Forex('EURUSD')
bars = ib.reqHistoricalData(
    contract, 
    endDateTime=end_time,
    durationStr='1 D',
    barSizeSetting='5 mins', 
    whatToShow='MIDPOINT', 
    useRTH=True)

# convert to pandas dataframe
df = util.df(bars)
print("Last completed candle:")
print(df.iloc[-1])  # Print the last row (most recent completed candle)

df.to_csv('EURUSD.csv', index=True)