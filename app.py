import os
import re
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DB_DIR = "chroma_db"
COLLECTION_NAME = "insurance_docs"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RETRIEVE_K = 10
FINAL_K = 4
MODEL_NAME = "gemini-2.5-flash"

# Required by bge models: queries need this exact instruction prefix,
# documents do not. This asymmetry is part of how the model was trained.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _tokenize(text):
    """Lowercase word tokens with light plural-stripping, so BM25 treats
    "plans" and "plan" as the same token instead of missing the match
    entirely (BM25 otherwise does pure exact-string matching)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w for w in words]

SYSTEM_PROMPT = """
You are a helpful customer service assistant for Harborview Health Insurance.
Use ONLY the provided context to answer.
If the context is insufficient, tell the customer to contact Harborview customer support for more information.
Use conversation history to understand follow-up questions like "it", "them", "that plan".
Be concise, accurate, and helpful.
"""


# ----------------------------
# LOAD RESOURCES
# ----------------------------
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # Pull the whole (small) corpus into memory once for BM25 keyword search.
    # Fine at this scale - if your doc set grows into the thousands of
    # chunks, move this to a proper BM25/Elasticsearch-style index instead.
    all_data = collection.get()
    all_chunks = all_data["documents"]
    all_sources = [m["source"] for m in all_data["metadatas"]]
    bm25 = BM25Okapi([_tokenize(c) for c in all_chunks])
    
    try:
      api_key = st.secrets["GEMINI_API_KEYOPENAI_API_KEY"]
    except:
      api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)

    return embed_model, reranker, collection, bm25, all_chunks, all_sources, gemini_client


# ----------------------------
# HISTORY-AWARE QUERY EXPANSION
# ----------------------------
def build_search_query(question):
    history = st.session_state.history[-4:]
    history_text = " ".join([msg["content"] for msg in history])
    return f"{question} {history_text}".strip()


# ----------------------------
# SEMANTIC (EMBEDDING) RETRIEVAL
# ----------------------------
def semantic_retrieve(query, embed_model, collection, k=RETRIEVE_K):
    query_embedding = embed_model.encode(
        [BGE_QUERY_PREFIX + query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas"],
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


# ----------------------------
# KEYWORD (BM25) RETRIEVAL
# ----------------------------
def keyword_retrieve(query, bm25, all_chunks, all_sources, k=RETRIEVE_K):
    scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    chunks = [all_chunks[i] for i in top_idx]
    sources = [all_sources[i] for i in top_idx]
    return chunks, sources


# ----------------------------
# HYBRID RETRIEVAL = SEMANTIC + KEYWORD, DE-DUPED
# ----------------------------
def hybrid_retrieve(query, embed_model, collection, bm25, all_chunks, all_sources, k=RETRIEVE_K):
    sem_chunks, sem_sources = semantic_retrieve(query, embed_model, collection, k=k)
    kw_chunks, kw_sources = keyword_retrieve(query, bm25, all_chunks, all_sources, k=k)

    combined = {}
    for chunk, source in zip(sem_chunks + kw_chunks, sem_sources + kw_sources):
        combined.setdefault(chunk, source)  # keeps first occurrence, de-dupes

    return list(combined.keys()), list(combined.values())


# ----------------------------
# RERANKING FUNCTION
# ----------------------------
def rerank(question, chunks, sources, reranker, final_k=FINAL_K):
    if not chunks:
        return [], []

    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks, sources), reverse=True, key=lambda x: x[0])

    reranked_chunks = [x[1] for x in ranked[:final_k]]
    reranked_sources = [x[2] for x in ranked[:final_k]]

    return reranked_chunks, reranked_sources


# ----------------------------
# GENERATION WITH MEMORY
# ----------------------------
def generate_answer(question, chunks, gemini_client):
    context = "\n\n---\n\n".join(chunks)

    chat_history = ""
    for msg in st.session_state.history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_history += f"{role}: {msg['content']}\n"

    prompt = f"""
Conversation history:
{chat_history}

Context documents:
{context}

Current question:
{question}
"""

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1200,
            ),
        )
        return response.text

    except ClientError as e:
        if "429" in str(e):
            return "API limit reached. Please try again later."
        return "API error occurred. Please try again later."

    except Exception:
        return "Unexpected error occurred. Please try again later."


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="HarborView")
st.title("Harbor - Insurance RAG Chatbot")
st.caption("Proof of concept. Answers are generated from a sample fictional company.")

embed_model, reranker, collection, bm25, all_chunks, all_sources, gemini_client = load_resources()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about your coverage, claims, or benefits...")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # HISTORY-AWARE QUERY casts a wider net for retrieval (so "what about
    # dental?" still pulls the right chunks), then RERANK scores against the
    # raw current question so the final picks stay precise.
    search_query = build_search_query(question)
    chunks, sources = hybrid_retrieve(
        search_query, embed_model, collection, bm25, all_chunks, all_sources
    )
    chunks, sources = rerank(question, chunks, sources, reranker)
    answer = generate_answer(question, chunks, gemini_client)

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Sources used"):
            for i, (src, chunk) in enumerate(zip(sources, chunks)):
                st.markdown(f"### Source {i+1}: {src}")
                st.code(chunk)

    st.session_state.history.append({"role": "assistant", "content": answer})