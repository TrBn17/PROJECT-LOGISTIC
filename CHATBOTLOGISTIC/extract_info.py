import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from chatstate import ChatState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Các field chuẩn có trong ChatState
REQUIRED_FIELDS = [
    "project_code",
    "country",
    "vendor",
    "freight_cost",
    "weight",
    "pack_price",
    "days_to_deliver",
    "pq_date",
]

PROMPT_TEMPLATE = """
You are a smart logistics assistant. Extract the following fields from the input text and return valid JSON:

Fields:
- project_code
- country
- vendor
- freight_cost
- weight
- pack_price
- days_to_deliver
- pq_date

Instructions:
- If any field is not mentioned, return null.
- Normalize field names to lowercase and use snake_case.
- For "deliver in 5 days", extract days_to_deliver = 5
- Ensure the output is strict valid JSON without any extra text.

Text:
\"\"\"{user_input}\"\"\"
"""

def extract_info(state: ChatState) -> ChatState:
    print("▶ [extract_info_node] Đang dùng GPT để bóc tách thông tin...")

    user_input = state.input_text.strip()
    if not user_input:
        state.extracted_info = {field: None for field in REQUIRED_FIELDS}
        state.extracted_info["raw_input"] = ""
        state.extracted_info["input_valid"] = False
        state.final_answer = "⚠️ Input đang trống, không thể trích xuất thông tin vận chuyển."
        return state

    try:
        prompt = PROMPT_TEMPLATE.format(user_input=user_input)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()

        # Remove markdown if any
        if content.startswith("```"):
            content = "\n".join(
                line for line in content.splitlines() if not line.strip().startswith("```")
            ).strip()

        extracted_raw = json.loads(content)

        # Chuẩn hóa và đảm bảo không field nào bị thiếu
        extracted_info = {}
        for field in REQUIRED_FIELDS:
            extracted_info[field] = extracted_raw.get(field, None)

        # Thêm các field đặc biệt
        extracted_info["raw_input"] = user_input
        extracted_info["input_valid"] = True

        state.extracted_info = extracted_info
        print("✅ [Extracted Info]:", json.dumps(extracted_info, indent=2))

    except Exception as e:
        print("❌ [Extract Error]:", str(e))
        state.extracted_info = {field: None for field in REQUIRED_FIELDS}
        state.extracted_info["raw_input"] = user_input
        state.extracted_info["input_valid"] = False
        state.final_answer = (
            "⚠️ Không trích xuất được thông tin từ input. "
            "Vui lòng nhập rõ ràng hơn hoặc thử lại sau."
        )
        state.error = f"[extract_info_node] {str(e)}"

    return state
