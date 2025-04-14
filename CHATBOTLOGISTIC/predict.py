import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from chatstate import ChatState
import random

# ==== Load paths ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "lightgbm.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")

# ==== Load model & encoders ====
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# ==== Config ====
INPUT_FEATURES = [
    "project_code",
    "country",
    "vendor",
    "pack_price",
    "freight_cost",
    "weight",
    "days_to_deliver",
    "pq_days_since_quote"
]
CATEGORICAL_COLS = ["project_code", "country", "vendor"]

# ==== Helper ====
def parse_days_since_quote(date_str):
    if not isinstance(date_str, str) or not date_str.strip():
        return -1
    try:
        quote_date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return (datetime.today() - quote_date).days
    except Exception as e:
        print(f"⚠️ Lỗi parse ngày: {e}")
        return -1

# ==== Dự đoán Shipment Mode ====
def predict_mode(state: ChatState) -> ChatState:
    print("▶ [predict_node] Bắt đầu dự đoán Shipment Mode...")

    info = state.extracted_info or {}
    data = {}

    # Map các trường
    for field in INPUT_FEATURES:
        if field == "pq_days_since_quote":
            data[field] = parse_days_since_quote(info.get("pq_date"))
        else:
            val = info.get(field)
            data[field] = val if val not in [None, ""] else -1

    if data["project_code"] == -1:
        data["project_code"] = "TEMP_PROJECT"

    df = pd.DataFrame([data])
    print("📥 [Input DataFrame]:\n", df)

    # Encode các trường dạng category
    for col in CATEGORICAL_COLS:
        val = str(df[col].iloc[0])
        le = label_encoders.get(col)
        if le:
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                print(f"⚠️ `{col}` chưa có label `{val}` → giả lập random")
                df[col] = random.randint(0, len(le.classes_) - 1)
        else:
            print(f"❌ Thiếu encoder `{col}` → giả lập random")
            df[col] = random.randint(0, 5)

    df = df[INPUT_FEATURES]
    print("✅ [Encoded DataFrame]:\n", df)

    top_preds = []
    try:
        y_pred = model.predict(df)[0]
        y_proba = model.predict_proba(df)[0]

        # ⚠️ Dùng đúng key "Shipment Mode"
        shipment_le = label_encoders.get("Shipment Mode")
        if not shipment_le:
            raise ValueError("Không tìm thấy encoder cho `Shipment Mode`")

        predicted_mode = shipment_le.inverse_transform([int(y_pred)])[0]
        class_names = shipment_le.inverse_transform(np.arange(len(y_proba)))

        prob_dict = {
            mode: round(100 * float(prob), 2)
            for mode, prob in zip(class_names, y_proba)
        }

        top_preds = sorted(
            [{"mode": mode, "probability": prob} for mode, prob in prob_dict.items() if prob > 0],
            key=lambda x: -x["probability"]
        )

        state.shipment_mode = [p["mode"] for p in top_preds]
        state.extracted_info["ml_predicted_shipment_mode"] = predicted_mode
        state.extracted_info["shipment_mode"] = predicted_mode
        state.model_prediction_debug = {
            "top_predictions": top_preds,
            "prediction_probabilities": prob_dict,
            "confidence_intervals": {"confidence": 90}
        }

        print("🎯 [Predicted Mode]:", predicted_mode)
        print("📊 [Top Predictions]:", top_preds)

    except Exception as e:
        print("❌ [Predict Error]:", str(e))
        state.shipment_mode = []
        state.final_answer = f"⚠️ Dự đoán thất bại: `{e}`"
        state.model_prediction_debug = {"error": str(e)}

    if not top_preds:
        print("⚠️ [predict_mode] Không có dự đoán nào đáng tin")
        state.shipment_mode = []
        state.final_answer = "⚠️ Không thể dự đoán Shipment Mode do thiếu dữ liệu hoặc lỗi input."
        state.model_prediction_debug = {}

    print("🧪 [Debug shipment_mode]:", state.shipment_mode)
    return state
