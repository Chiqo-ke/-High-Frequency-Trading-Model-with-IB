# trading-data-redis

This project integrates Redis as an in-memory database to store data fetched from TradingView. It provides a framework for fetching trading data and storing it efficiently for quick access.

## Project Structure

```
trading-data-redis
├── src
│   ├── data_fetcher
│   │   ├── __init__.py
│   │   ├── tradingview.py
│   │   └── redis_store.py
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── logging.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_tradingview.py
│   └── test_redis_store.py
├── data
│   └── .gitkeep
├── logs
│   └── .gitkeep
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trading-data-redis
   ```

2. **Install dependencies:**
   Make sure you have Python and pip installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Redis:**
   Update the `src/config/settings.py` file with your Redis connection parameters.

4. **Run the application:**
   You can run the application using:
   ```bash
   python src/main.py
   ```

5. **Running with Docker:**
   To run the application using Docker, use the following command:
   ```bash
   docker-compose up
   ```

## Usage

- The application fetches trading data from TradingView and stores it in Redis.
- You can modify the `src/main.py` file to customize the symbols and intervals for data fetching.

## Testing

To run the tests, use:
```bash
pytest tests/
```

## Logging

Logs are stored in the `logs` directory. You can configure logging settings in `src/utils/logging.py`.

## License

This project is licensed under the MIT License.