from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ChatState:
    user_id: str
    input_text: str

    extracted_info: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    shipment_mode: list = field(default_factory=list)
    prediction_prob: Optional[float] = None
    label_probs: dict = field(default_factory=dict)
    final_answer: Optional[str] = None

    # ✅ Hỗ trợ mode hỗ trợ sau tư vấn
    support_mode: bool = False
    support_count: int = 0
