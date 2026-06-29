import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
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
TOP_K = 6
MODEL_NAME = "gemini-2.5-flash"

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
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)

    return embed_model, collection, gemini_client


# ----------------------------
# HISTORY-AWARE QUERY EXPANSION
# ----------------------------
def build_search_query(question):
    history = st.session_state.history[-4:]
    history_text = " ".join([msg["content"] for msg in history])
    return f"{question} {history_text}"

# ----------------------------
# RETRIEVAL
# ----------------------------
def retrieve(question, embed_model, collection, k=TOP_K):
    query_embedding = embed_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas"]
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    return chunks, sources

# ----------------------------
# GENERATION WITH MEMORY
# ----------------------------
def generate_answer(question, chunks, gemini_client):
    context = "\n\n---\n\n".join(chunks)

    # Build chat history
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
            return "⚠️ API limit reached. Please try again later."
        return "⚠️ API error occurred. Please try again later."

    except Exception:
        return "⚠️ Unexpected error occurred. Please try again later."


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="HarborView")
st.title("Harbor - Insurance RAG Chatbot")
st.caption("Proof of concept. Answers are generated from a sample fictional company.")

embed_model, collection, gemini_client = load_resources()

if "history" not in st.session_state:
    st.session_state.history = []

# Show chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about your coverage, claims, or benefits...")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # HISTORY-AWARE RETRIEVAL INPUT
    search_query = build_search_query(question)
    chunks, sources = retrieve(search_query, embed_model, collection)

    answer = generate_answer(question, chunks, gemini_client)

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Sources used"):
            for i, (src, chunk) in enumerate(zip(sources, chunks)):
                st.markdown(f"### Source {i+1}: {src}")
                st.code(chunk)

    st.session_state.history.append({"role": "assistant", "content": answer})

    