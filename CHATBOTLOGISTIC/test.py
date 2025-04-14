from chatstate import ChatState
from graph import build_graph

if __name__ == "__main__":
    print("🚀 Running test graph...")

    # Câu test mẫu
    input_text = (
        "We need to ship 820kg of equipment from ABBOTT GmbH to Vietnam. "
        "The pack price is $18.5 and the freight cost is $2300. "
        "Project code is 108-VN-T01. PQ First Sent to Client Date is 2024-03-12. "
        "Expected delivery in 7 days."
    )

    # Khởi tạo trạng thái
    state = ChatState(user_id="test_user", input_text=input_text)

    # Build và chạy graph
    graph_app = build_graph()
    result = graph_app.invoke(state)

    # In kết quả
    print("\n🧠 FINAL ANSWER:")
    print(result.get("final_answer") or "⚠️ No final answer returned.")
