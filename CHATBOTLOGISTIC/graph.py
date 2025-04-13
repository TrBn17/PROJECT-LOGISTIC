from langgraph.graph import StateGraph, END
from chatstate import ChatState
from input import process_input
from extract_info import extract_info
from encode_features import encode_features
from predict import predict_mode
from call_gpt import call_gpt
from langchain_core.runnables import RunnableLambda

def log_node(name: str, func) -> RunnableLambda:
    def wrapper(state: ChatState) -> ChatState:
        print(f"▶️ [DEBUG] Bắt đầu node: {name}")
        result = func(state)
        print(f"✅ [DEBUG] Kết thúc node: {name}")
        return result
    return RunnableLambda(wrapper)

def build_graph():
    workflow = StateGraph(ChatState)

    workflow.add_node("input", log_node("input", process_input))
    workflow.add_node("extract_info", log_node("extract_info", extract_info))
    workflow.add_node("encode_features", log_node("encode_features", encode_features))
    workflow.add_node("predict", log_node("predict", predict_mode))
    workflow.add_node("call_model", log_node("call_model", call_gpt))

    workflow.set_entry_point("input")
    workflow.add_edge("input", "extract_info")
    workflow.add_edge("extract_info", "encode_features")
    workflow.add_edge("encode_features", "predict")
    workflow.add_edge("predict", "call_model")
    workflow.add_edge("call_model", END)

    return workflow.compile()
