import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableLambda
from chatstate import ChatState
import joblib

# Load model + encoders
mapie_model = joblib.load("mapie_model_lgb.pkl")
label_encoders = joblib.load("label_encoder.pkl")
shipment_encoder = label_encoders["Shipment Mode"]

# Cột đầu vào đúng thứ tự
FEATURE_COLUMNS = [
    'DaysToDeliver',
    'Project Code',
    'Country',
    'Pack Price',
    'Vendor',
    'Freight Cost (USD)',
    'Weight (Kilograms)',
    'PQ First Sent to Client Date'
]

# Phương thức fallback (nếu model không trả kết quả)
FALLBACK_METHOD = "Standard International Shipping"

def predict_mode(state: ChatState) -> ChatState:
    print("▶ [predict_node] Đang xử lý dự đoán MAPIE...")
    features = state.features or {}

    # Format input đúng định dạng DataFrame
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    X = X.fillna(-1)

    try:
        # Dự đoán tập nhãn với độ tin cậy 90% từ MAPIE
        prediction_set = mapie_model.predict(X, alpha=0.3)
        predicted_labels = prediction_set[0]

        if len(predicted_labels) == 0:
            raise ValueError("Không có nhãn nào được dự đoán.")

        # Convert index thành tên phương thức
        label_names = shipment_encoder.inverse_transform(predicted_labels)

        # Lấy xác suất thực tế nếu model gốc hỗ trợ
        try:
            base_model = mapie_model.estimator_
            proba = base_model.predict_proba(X)[0]

            # Lấy top-n label có xác suất cao nhất (ví dụ n=3)
            top_n = 3
            top_indices = np.argsort(proba)[-top_n:][::-1]
            label_names = shipment_encoder.inverse_transform(top_indices)

            label_probs = {
                shipment_encoder.inverse_transform([i])[0]: float(proba[i])
                for i in top_indices
}
        except:
            label_probs = {label: None for label in label_names}

        # Tạo chuỗi hiển thị cho GPT hoặc logging
        label_str_list = []
        for label in label_names:
            prob = label_probs[label]
            if prob is not None:
                label_str_list.append(f"**{label}** (xác suất ~{round(prob * 100, 1)}%)")
            else:
                label_str_list.append(f"**{label}**")

        label_str = ", ".join(label_str_list)

        # Gán vào state đầy đủ
        state.shipment_mode = list(label_names)
        state.prediction_prob = max(label_probs.values()) if any(label_probs.values()) else None
        state.label_probs = label_probs  # <-- KEY LINE giúp GPT hiểu
        state.final_answer = (
            f"Mô hình MAPIE dự đoán các phương thức vận chuyển phù hợp là: {label_str}. "
            f"Đây là những phương án có khả năng cao nằm trong 90% độ tin cậy của mô hình.\n\n"
            f"📦 Trong trường hợp ngoại lệ (10%), phương thức dự phòng được đề xuất là: **{FALLBACK_METHOD}**."
        )

    except Exception as e:
        state.shipment_mode = [FALLBACK_METHOD]
        state.prediction_prob = 0.0
        state.label_probs = {}
        state.final_answer = (
            f"⚠️ Không thể đưa ra dự đoán do lỗi: {str(e)}\n"
            f"Gợi ý sử dụng phương thức dự phòng: **{FALLBACK_METHOD}**."
        )
    print(f"🧠 MAPIE trả: {state.shipment_mode}")
    print(f"📊 Xác suất: {state.label_probs}")

    return state
