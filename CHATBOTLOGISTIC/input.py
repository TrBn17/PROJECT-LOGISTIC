from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from chatstate import ChatState


class InputInfo(TypedDict):
    user_id: str
    input_text: str


def process_input(state: ChatState) -> ChatState:
    print("▶ [input_node] Đang xử lý đầu vào...")
    print(f"[INPUT NODE] User {state.user_id} sent: {state.input_text}")

    # Chuẩn hóa input
    user_text = (state.input_text or "").strip().lower()

    if not user_text:
        print("⚠️ [INPUT NODE] Input rỗng, không thể xử lý.")
        state.extracted_info = {
            "raw_input": "",
            "project_code": "",
            "country": "",
            "vendor": "",
            "freight_cost": -1.0,
            "weight": -1.0,
            "pack_price": -1.0,
            "days_to_deliver": -1,
            "pq_date": "",
            "input_valid": False,
        }
        return state

    # Gợi ý tự động từ text input
    def extract_country(text: str) -> str:
        for country in ["vietnam", "china", "india", "indonesia", "united states", "usa", "us", "germany", "france"]:
            if country in text:
                return country.title()
        return ""

    def extract_vendor(text: str) -> str:
        for vendor in ["pfizer", "roche", "astrazeneca", "gsk", "novartis"]:
            if vendor in text:
                return vendor.title()
        return ""

    state.extracted_info = {
        "raw_input": state.input_text.strip(),
        "project_code": "TEMP_PROJECT",  # Có thể gán mặc định, hoặc để trống nếu chưa rõ
        "country": extract_country(user_text),
        "vendor": extract_vendor(user_text),
        "freight_cost": -1.0,
        "weight": -1.0,
        "pack_price": -1.0,
        "days_to_deliver": -1,
        "pq_date": "",  # Có thể dùng NLP để trích ngày nếu cần
        "input_valid": True,
    }

    return state
