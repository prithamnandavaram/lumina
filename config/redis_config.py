import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL", "redis://localhost:6380")

queue = Queue(connection=Redis.from_url(redis_url))
