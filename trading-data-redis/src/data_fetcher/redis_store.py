# redis_store.py
import redis
import json
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class RedisStore:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize the Redis store with connection parameters."""
        try:
            self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)
            logging.info("Connected to Redis")
        except Exception as e:
            logging.error(f"Error connecting to Redis: {e}")
            raise
            
    def save_data(self, key, data):
        """Save data to Redis with proper datetime handling"""
        try:
            if isinstance(data, pd.DataFrame):
                # Convert datetime columns to ISO format strings
                data_dict = data.copy()
                for column in data_dict.select_dtypes(include=['datetime64[ns]']).columns:
                    data_dict[column] = data_dict[column].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                json_data = data_dict.to_json(orient='records', date_format='iso')
            else:
                # Handle non-DataFrame data
                json_data = json.dumps(data, default=self._datetime_handler)
            
            self.client.set(key, json_data)
            logger.info(f"Successfully saved data to Redis with key: {key}")
        except Exception as e:
            logger.error(f"Error saving data to Redis: {e}")
            
    def get_data(self, key):
        """Retrieve DataFrame from Redis."""
        try:
            data = self.client.get(key)
            if data:
                df = pd.read_json(data, orient='records')
                logging.info(f"Data retrieved from Redis with key: {key}")
                return df
            else:
                logging.warning(f"No data found in Redis for key: {key}")
                return None
        except Exception as e:
            logging.error(f"Error retrieving data from Redis: {e}")
            return None

    def _datetime_handler(self, obj):
        """Handle datetime serialization for JSON"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")