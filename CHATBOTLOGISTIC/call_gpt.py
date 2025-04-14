import os
from openai import OpenAI
from dotenv import load_dotenv
from chatstate import ChatState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt(state: ChatState) -> ChatState:
    print("▶ [call_model_node] Gọi GPT để phân tích dự đoán...")

    # ==== Debug info ====
    print("📦 [DEBUG] extracted_info:", state.extracted_info)
    print("📦 [DEBUG] model_prediction_debug:", state.model_prediction_debug)
    print("📦 [DEBUG] shipment_mode:", getattr(state, "shipment_mode", []))
    print("📦 [DEBUG] context len:", len(state.context or ""))
    print("📦 [DEBUG] support_count:", getattr(state, "support_count", 0))

    # ==== Validate ML prediction ====
    prediction = state.model_prediction_debug or {}
    top_preds = prediction.get("top_predictions", [])

    if not isinstance(top_preds, list) or not top_preds:
        print("❌ [DEBUG] top_predictions rỗng hoặc sai định dạng.")
        state.final_answer = "⚠️ Không thể đưa ra gợi ý do thiếu dữ liệu dự đoán từ AI."
        return state

    try:
        label_probs = {
            p.get("mode", "?"): p.get("probability", 0) / 100
            for p in top_preds if isinstance(p, dict) and p.get("probability", 0) > 0
        }
    except Exception as e:
        state.final_answer = f"⚠️ Dữ liệu xác suất không hợp lệ: `{e}`"
        return state

    if not label_probs:
        state.final_answer = "⚠️ Không thể đưa ra gợi ý do dữ liệu xác suất không đáng tin cậy."
        return state

    # ==== Summary ====
    info = state.extracted_info or {}
    summary = f"""
The customer needs to ship *{info.get('weight', '?')}kg* to *{info.get('country', '?')}*.
Packing price: *${info.get('pack_price', '?')}*, Freight cost: *${info.get('freight_cost', '?')}*.
Vendor: *{info.get('vendor', '?')}*, Project Code: *{info.get('project_code', '?')}*, PQ Date: *{info.get('pq_date', '?')}*.
""".strip()

    mode_lines = "\n".join(
        f"- *{mode}* ({round(prob * 100, 1)}%)"
        for mode, prob in label_probs.items()
    )

    context_part = ""
    if getattr(state, "support_count", 0) > 0 and state.context:
        context_part = f"\n\n📚 *Legal reference (if applicable):*\n{state.context.strip()}"

    # ==== Prompt for GPT ====
    prompt = f"""
You are a logistics advisor AI. Analyze this shipment request:

{summary}

🎯 Predicted shipment mode(s) based on internal AI model:
{mode_lines}

🛑 *Use only the shipment modes listed above.*{context_part}

Tasks:
1. Analyze pros and cons of each shipment mode.
2. Recommend the most suitable one based on cost, delivery time, and safety.
3. If legal context is present, apply it to support your reasoning.

Respond concisely and professionally.
""".strip()

    print("📝 [DEBUG] GPT prompt preview:\n", prompt[:500], "..." if len(prompt) > 500 else "")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600
        )

        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("GPT trả về nội dung rỗng.")

        state.final_answer = content
        state.support_mode = True
        state.support_count = getattr(state, "support_count", 0) + 1

        print("✅ [GPT OK] Final answer đã được gán vào state.")

    except Exception as e:
        print("❌ [GPT ERROR]:", str(e))
        state.final_answer = f"⚠️ GPT gặp lỗi: `{e}`"

    return state
