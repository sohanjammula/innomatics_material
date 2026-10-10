
import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv()

redis_url = os.environ.get("REDIS_URL")
# session_id = "456"
# history = RedisChatMessageHistory(
#         session_id=session_id,
#         url=redis_url,
#         key_prefix="chat_history:",
#     )
# messages = history.messages

# for i, message in enumerate(messages):
#      print(f"Message {i + 1}: {type(message).__name__} - {message.content}")

import redis

r = redis.Redis(
    host='redis-14921.crce220.us-east-1-4.ec2.cloud.redislabs.com',
    port=14921,
    decode_responses=True,
    username="default",
    password=os.environ.get("REDIS_PASSWORD"),
)

all_keys = r.keys('*')
import json
print(all_keys)
# for each_key in all_keys:
#     messages = r.lrange(each_key, 0, -1)
#     print(f"Messages for key '{each_key}':")
#     for i, message in enumerate(messages):
#         message = json.loads(message)
#         print(f"Message {i + 1}: {message.get('type')} - {message["data"].get('content')}")
