from data_fetcher.tradingview import TradingViewDataFetcher
from data_fetcher.redis_store import RedisStore
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB

def main():
    # Initialize Redis store
    redis_store = RedisStore(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    
    # Initialize TradingView data fetcher
    fetcher = TradingViewDataFetcher()
    
    # Example symbol and exchange
    symbol = "XAUUSD"
    exchange = "OANDA"
    
    # Fetch historical data
    h1_data = fetcher.fetch_historical_data(symbol, exchange, 'H1', days_back=10)
    m5_data = fetcher.fetch_historical_data(symbol, exchange, 'M5', days_back=10)
    
    # Store fetched data in Redis
    if h1_data is not None:
        redis_store.save_data('h1_data', h1_data.to_dict(orient='records'))
    
    if m5_data is not None:
        redis_store.save_data('m5_data', m5_data.to_dict(orient='records'))

if __name__ == "__main__":
    main()