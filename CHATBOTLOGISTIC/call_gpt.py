import os
from openai import OpenAI
from langchain_core.runnables import RunnableLambda
from chatstate import ChatState
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt(state: ChatState) -> ChatState:
    print("▶ [call_model_node] GPT đang được gọi để giải thích...")
    info = state.extracted_info or {}

    modes = state.shipment_mode if isinstance(state.shipment_mode, list) else [state.shipment_mode]
    label_probs = getattr(state, "label_probs", {})

    # Format các phương thức được AI đề xuất (thêm xác suất nếu có)
    mode_lines = "\n".join(
    f"- {mode} ({round(label_probs[mode]*100, 1)}%)" if label_probs.get(mode) is not None else f"- {mode}"
    for mode in modes
)
    order_summary = f"""
Quý khách cần gửi {info.get('weight')}kg linh kiện điện tử từ {info.get('origin_country')} đến {info.get('destination_country')}.
Giá mỗi pack là {info.get('pack_price')} USD, phí vận chuyển khoảng {info.get('freight_cost')} USD.
Dự án mã {info.get('project_code')} từ đối tác {info.get('vendor')}, dự kiến gửi vào ngày {info.get('pq_date')}.
"""
    prompt = f"""
Bạn là một chuyên gia tư vấn vận chuyển thông minh. Dưới đây là thông tin đơn hàng:

{order_summary}

Kết quả từ mô hình AI MAPIE: các phương thức vận chuyển được đề xuất là:
{mode_lines}

❗Lưu ý: Đây là toàn bộ kết quả từ AI, không được bịa đặt hay phân tích phương thức khác.

Yêu cầu:
1. Phân tích từng phương thức nêu trên, **liên hệ trực tiếp với đơn hàng này**, nêu rõ ưu điểm và nhược điểm.
2. So sánh các phương thức nếu có nhiều.
3. Gợi ý lựa chọn phù hợp nhất tùy theo mục tiêu (giao nhanh, tiết kiệm chi phí, an toàn).

Trình bày chuyên nghiệp, ngắn gọn, dễ hiểu.

Trân trọng,
[Chuyên viên Logistic]
"""



    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200,
        top_p=1.0
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200,
        top_p=1.0
    )

    # ✅ Phản hồi GPT chính
    state.final_answer = response.choices[0].message.content.strip()

    # ✅ Thêm phần support tiếp theo
    state.final_answer += (
        "\n\n🤖 Quý khách có cần tôi hỗ trợ thêm về giá vận chuyển, thời gian giao hàng, "
        "quy trình thủ tục hải quan, hoặc thông tin về nhà cung cấp không ạ? "
        "Vui lòng nhắn tin để tiếp tục tư vấn."
    )
    state.support_mode = True
    state.support_count = 0
    return state
