
import redis
import os
from dotenv import load_dotenv

load_dotenv()

class RedisClient():
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT',6379))
        self.redis_db = int(os.getenv('REDIS_DB', 0))

        print(f"Redis Connection Details:")
        print(f"Host: '{self.redis_host}' (type: {type(self.redis_host)})")
        print(f"Port: {self.redis_port} (type: {type(self.redis_port)})")
        print(f"DB: {self.redis_db}")

        self.client = None

    def connect(self):
        try:
            self.client = redis.Redis(
                host = self.redis_host,
                port = self.redis_port,
                db = self.redis_db,
                decode_responses = True
            )

            #Test connection
            self.client.ping()
            print("Connected to Redis server successfully!")

            return self.client
        
        except redis.ConnectionError as e:
            print(f"Redis conenction error:{e}")
            raise

    def get_client(self):
        if not self.client:
            self.connect()
        return self.client
    

redis_client = RedisClient()