import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_data_fetcher.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)