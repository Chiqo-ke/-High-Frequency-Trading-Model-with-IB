import unittest
from src.data_fetcher.tradingview import TradingViewDataFetcher

class TestTradingViewDataFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = TradingViewDataFetcher()

    def test_initialization(self):
        self.assertIsNotNone(self.fetcher.tv)

    def test_fetch_historical_data(self):
        symbol = 'BTCUSDT'
        exchange = 'BINANCE'
        interval = 'H1'
        data = self.fetcher.fetch_historical_data(symbol, exchange, interval, days_back=1)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_update_data(self):
        symbol = 'BTCUSDT'
        exchange = 'BINANCE'
        interval = 'H1'
        self.fetcher.fetch_historical_data(symbol, exchange, interval, days_back=1)
        updated_data = self.fetcher.update_data(symbol, exchange, interval)
        self.assertIsNotNone(updated_data)

if __name__ == '__main__':
    unittest.main()