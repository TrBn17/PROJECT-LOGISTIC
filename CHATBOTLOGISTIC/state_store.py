# state_store.py
from cachetools import TTLCache

# Tạo cache lưu trạng thái user, mặc định 1 tiếng
user_state_store = TTLCache(maxsize=10000, ttl=3600)

def reset_user_state(user_id: str):
    if user_id in user_state_store:
        del user_state_store[user_id]

def save_user_state(user_id: str, state):
    user_state_store[user_id] = state

def get_user_state(user_id: str):
    return user_state_store.get(user_id)
