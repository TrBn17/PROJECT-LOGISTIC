# extract_info.py
import re
from langchain_core.runnables import RunnableLambda
from chatstate import ChatState

def extract_info(state: ChatState) -> ChatState:
    print("▶ [extract_info_node] Đang xử lý trích xuất thông tin...")
    text = state.input_text.lower()
    info = {}

    # 🧠 Helper: extract string sau từ khóa
    def extract_text_after(keyword, text):
        match = re.search(rf"{keyword}\s*:?[\s]*([\w\s\-,]+)", text)
        return match.group(1).strip().title() if match else None

    # 🧠 Helper: extract float từ các từ khóa liên quan
    def extract_float(keywords, text):
        for key in keywords:
            match = re.search(rf"{key}\s*:?[\s$]*([\d\.]+)", text)
            if match:
                return float(match.group(1))
        return None

    # 🧠 Helper: extract int
    def extract_int(keywords, text):
        for key in keywords:
            match = re.search(rf"{key}\s*:?[\s]*([\d]+)", text)
            if match:
                return int(match.group(1))
        return None

    # 🧠 Helper: extract date
    def extract_date(text):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        return match.group(1) if match else None

    # 📦 Trích thông tin đầu vào
    info["origin_country"] = extract_text_after(r"(from|từ)", text)
    info["destination_country"] = extract_text_after(r"(to|đến|tới)", text)

    info["weight"] = extract_float(["weight", "nặng", "trọng lượng"], text)
    info["pack_price"] = extract_float(["pack price", "giá mỗi pack", "giá"], text)
    info["freight_cost"] = extract_float(["freight", "vận chuyển", "shipping", "phí"], text)

    info["project_code"] = extract_int(["project", "mã dự án"], text)

    info["vendor"] = extract_text_after("vendor", text)
    info["pq_date"] = extract_date(text)

    # ✅ Gán vào state
    state.extracted_info = info
    return state

extract_info_node: RunnableLambda = RunnableLambda(extract_info)
