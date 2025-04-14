from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ChatState:
    user_id: str
    input_text: str

    extracted_info: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    shipment_mode: List[str] = field(default_factory=list)
    prediction_prob: Optional[float] = None
    label_probs: dict = field(default_factory=dict)
    final_answer: Optional[str] = None

    # ✅ Dự đoán chi tiết từ model
    model_prediction_debug: dict = field(default_factory=dict)

    # ✅ Trợ lý có đang hỗ trợ (gọi GPT sau tư vấn lần 1 trở đi)
    support_mode: bool = False
    support_count: int = 0

    # ✅ Tin nhắn thứ mấy trong session (dùng để kiểm tra gọi GPT2 hay chưa)
    message_count: int = 1  # 👈 THÊM DÒNG NÀY

    # ✅ Truy vấn vector hóa (để dùng sau)
    context: str = ""
    vector_matches: List[Dict] = field(default_factory=list)

    # ✅ Tracking lỗi và log debug
    error: Optional[str] = None
    debug_log: List[str] = field(default_factory=list)

    # ✅ Optional: theo dõi session/timestamp
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
