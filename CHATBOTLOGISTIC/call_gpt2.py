import os
from openai import OpenAI
from dotenv import load_dotenv
from chatstate import ChatState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt2(state: ChatState) -> ChatState:
    print("▶ [call_gpt2_node] Gọi GPT để phân tích luật từ vector search...")

    if not state.context:
        print("⚠️ [call_gpt2_node] Không có context để phân tích luật.")
        state.final_answer = "⚠️ Không tìm thấy thông tin luật phù hợp để tư vấn."
        return state

    shipment_mode = state.extracted_info.get("shipment_mode", None)
    if not shipment_mode:
        state.final_answer = "⚠️ Không xác định được shipment mode để áp dụng luật."
        return state

    if isinstance(shipment_mode, list):
        shipment_mode_str = ", ".join([m.upper() for m in shipment_mode])
    else:
        shipment_mode_str = str(shipment_mode).upper()

    prompt = f"""
You are a logistics legal advisor AI. Analyze the following legal context related to the shipment mode **{shipment_mode_str}**:

{state.context}

Tasks:
1. Summarize the key legal obligations and constraints for using this shipment mode.
2. Identify any risks or legal requirements the customer must be aware of.
3. Provide recommendations to stay compliant and avoid issues.

Respond in 3 sections with clear headings:
### Legal Obligations
...
### Risks and Legal Requirements
...
### Compliance Recommendations
...
""".strip()

    print("🧠 [DEBUG] GPT2 Prompt:\n", prompt[:500], "..." if len(prompt) > 500 else "")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal compliance assistant specializing in logistics regulations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("GPT trả về nội dung rỗng.")

        state.final_answer = content
        print("✅ [call_gpt2] GPT đã phản hồi phân tích luật thành công.")

    except Exception as e:
        print("❌ [GPT ERROR - call_gpt2]:", str(e))
        state.final_answer = f"⚠️ GPT gặp lỗi khi tư vấn luật: `{e}`"

    return state
