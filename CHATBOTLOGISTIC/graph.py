from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from chatstate import ChatState
from input import process_input
from extract_info import extract_info
from predict import predict_mode
from call_gpt import call_gpt
from vector_search import search_vector
from call_gpt2 import call_gpt2

def log_node(name: str, func) -> RunnableLambda:
    def wrapper(state: ChatState) -> ChatState:
        print(f"▶️ [{name}_node] Gọi {name}...")
        result = func(state)
        print(f"✅ [{name}_node] Done.")
        return result
    return RunnableLambda(wrapper)

def maybe_call_model(state: ChatState) -> ChatState:
    print(f"🧪 [Flow] message_count = {state.message_count}")
    if state.message_count == 1:
        print("✅ [Flow] message_count == 1 → GỌI GPT1")
        return call_gpt(state)
    else:
        print("⛔ [Flow] message_count > 1 → BỎ QUA GPT1")
        return state

def run_vector_search_if_needed(state: ChatState) -> ChatState:
    if state.support_count > 0:
        print("✅ [Flow] support_count > 0 → GỌI vector_search")
        return search_vector(state)
    else:
        print("⛔ [Flow] support_count = 0 → DỪNG")
        state.final_answer = "✅ We've completed your request!"
        return state

def run_gpt2_if_needed(state: ChatState) -> ChatState:
    if state.message_count >= 2:
        print("✅ [Flow] message_count >= 2 → GỌI GPT2")
        return call_gpt2(state)
    else:
        print("⛔ [Flow] message_count < 2 → DỪNG")
        return state

def build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("input", log_node("input", process_input))
    workflow.add_node("extract_info", log_node("extract_info", extract_info))
    workflow.add_node("predict", log_node("predict", predict_mode))
    workflow.add_node("maybe_call_model", RunnableLambda(maybe_call_model))
    workflow.add_node("maybe_vector_search", RunnableLambda(run_vector_search_if_needed))
    workflow.add_node("maybe_gpt2", RunnableLambda(run_gpt2_if_needed))

    workflow.set_entry_point("input")
    workflow.add_edge("input", "extract_info")
    workflow.add_edge("extract_info", "predict")
    workflow.add_edge("predict", "maybe_call_model")
    workflow.add_edge("maybe_call_model", "maybe_vector_search")
    workflow.add_edge("maybe_vector_search", "maybe_gpt2")
    workflow.add_edge("maybe_gpt2", END)

    return workflow.compile()
