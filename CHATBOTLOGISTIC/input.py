# input.py
from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from chatstate import ChatState

class InputInfo(TypedDict):
    user_id: str
    input_text: str

def process_input(state: ChatState) -> ChatState:
    print("▶ [input_node] Đang xử lý đầu vào...")
    print(f"[INPUT NODE] User {state.user_id} sent: {state.input_text}")

    # Tạm thời gán thông tin vào extracted_info (node sau sẽ extract)
    state.extracted_info = {
        "raw_input": state.input_text.strip(),
        # placeholder cho các field khác
        "origin_country": "Vietnam",
        "destination_country": None,  # default để node sau extract
        "country": None,
        "weight": None,
        "vendor": None,
        "freight_cost": None,
        "pack_price": None,
        "days_to_deliver": None,
        "pq_date": None,
        "project_code": None,
    }

    return state
