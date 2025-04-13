# encode_features.py

from langchain_core.runnables import RunnableLambda
from chatstate import ChatState
import joblib

label_encoders = joblib.load("label_encoder.pkl")

# Các cột cần encode
CATEGORICAL_COLUMNS = [
    'Project Code',
    'Country',
    'Destination Country',  # 🆕 Thêm điểm đến
    'Vendor',
    'PQ First Sent to Client Date'
]

# Các cột đầu vào theo đúng thứ tự mô hình đã huấn luyện
FEATURE_COLUMNS = [
    'DaysToDeliver',
    'Project Code',
    'Country',
    'Destination Country',  # 🆕 Thêm điểm đến
    'Pack Price',
    'Vendor',
    'Freight Cost (USD)',
    'Weight (Kilograms)',
    'PQ First Sent to Client Date'
]

def normalize_key(col: str) -> str:
    return col.lower().replace(" (usd)", "").replace(" ", "_")

def encode_features(state: ChatState) -> ChatState:
    print("▶ [encode_features_node] Đang xử lý mã hóa đặc trưng...")
    info = state.extracted_info or {}
    features = {}

    for col in FEATURE_COLUMNS:
        key = normalize_key(col)
        if col in CATEGORICAL_COLUMNS:
            encoder = label_encoders.get(col)
            raw_val = info.get(key)
            try:
                features[col] = encoder.transform([raw_val])[0] if encoder and raw_val is not None else -1
            except:
                features[col] = -1
        else:
            features[col] = info.get(key, 0)

    state.features = features
    return state

encode_features_node: RunnableLambda = RunnableLambda(encode_features)
