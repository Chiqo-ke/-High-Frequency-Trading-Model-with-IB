import unittest
from src.data_fetcher.redis_store import RedisStore

class TestRedisStore(unittest.TestCase):
    def setUp(self):
        self.redis_store = RedisStore()
        self.test_key = 'test_key'
        self.test_value = 'test_value'
        self.redis_store.connect()

    def tearDown(self):
        self.redis_store.delete(self.test_key)

    def test_save_and_retrieve_data(self):
        self.redis_store.save(self.test_key, self.test_value)
        retrieved_value = self.redis_store.retrieve(self.test_key)
        self.assertEqual(retrieved_value, self.test_value)

    def test_delete_data(self):
        self.redis_store.save(self.test_key, self.test_value)
        self.redis_store.delete(self.test_key)
        retrieved_value = self.redis_store.retrieve(self.test_key)
        self.assertIsNone(retrieved_value)

if __name__ == '__main__':
    unittest.main()