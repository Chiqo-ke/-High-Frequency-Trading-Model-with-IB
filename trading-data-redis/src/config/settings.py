# Configuration settings for the application
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
# Redis connection settings
REDIS_CONFIG = {
    'host': 'redis',  # or 'redis' if using Docker
    'port': 6379,
    'db': 0,
    'decode_responses': True
}

# Logging settings
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOGGING_FILE = 'data_fetcher.log'