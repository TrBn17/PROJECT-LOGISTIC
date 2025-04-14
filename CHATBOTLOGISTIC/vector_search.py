import os
import chromadb
import openai
from dotenv import load_dotenv
from chatstate import ChatState

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_HOST = "local_host"
CHROMA_PORT = "8000"
COLLECTION_NAME = "usa_transportation_law"
MODEL_NAME = "text-embedding-3-large"

def get_query_embedding(query: str):
    try:
        res = openai.embeddings.create(
            input=query,
            model=MODEL_NAME
        )
        return res.data[0].embedding
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return None

# ✅ Extract mode từ input nếu có
def extract_mode(text: str) -> str:
    text = text.lower()
    if "air" in text:
        return "air"
    elif "ocean" in text or "sea" in text:
        return "ocean"
    elif "truck" in text:
        return "truck"
    elif "charter" in text:
        return "air charter"
    return ""

def search_vector(state: ChatState) -> ChatState:
    print("🔍 [search_vector_node] Performing vector search...")

    embedding = get_query_embedding(state.input_text)
    if not embedding:
        state.context = ""
        state.vector_matches = []
        return state

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(COLLECTION_NAME)

    # ✅ Tự động apply filter nếu input rõ mode
    mode_filter = extract_mode(state.input_text)
    if mode_filter:
        print(f"🔍 [filter] mode = {mode_filter}")
        results = collection.query(
            query_embeddings=[embedding],
            n_results=4,
            where={"mode": mode_filter}
        )
    else:
        print("🔍 [filter] No mode detected → querying all")
        results = collection.query(
            query_embeddings=[embedding],
            n_results=4
        )

    if not results or not results.get("documents"):
        state.context = ""
        state.vector_matches = []
        return state

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    matches = []

    for doc, meta in zip(docs, metas):
        mode = meta.get("mode", "unknown").upper()
        context_parts.append(f"[{mode}]\n{doc.strip()}")
        matches.append({
            "text": doc.strip(),
            "mode": meta.get("mode", "unknown"),
            "length": meta.get("length", len(doc.strip()))
        })

    state.context = "\n\n---\n\n".join(context_parts)
    state.vector_matches = matches

    print(f"✅ [search_vector_node] Found {len(matches)} matches.")
    return state
