# Harbor — RAG-Based Insurance Document Chatbot
### 🔗 [Live Demo](https://harbor-insurance.streamlit.app/) 
A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
a fictional health insurance company's plans, coverage, and claims process —
grounded entirely in a small set of sample documents, with conversational
memory and source citations.

Built as a learning project / proof of concept to explore the full RAG
pipeline: chunking, hybrid retrieval, reranking, and grounded generation.

<img width="1919" height="865" alt="Chatbot UI image" src="https://github.com/user-attachments/assets/277b20f0-f705-4240-9a75-4f3a001c59ff" />


## Features

- **Hybrid retrieval** — combines semantic (embedding) search with BM25
  keyword search, so both "meaning" matches and exact terms (plan names,
  dollar amounts) get surfaced
- **Cross-encoder reranking** — a second-stage model reorders retrieved
  chunks for relevance before they reach the LLM
- **Conversation memory** — follow-up questions like *"what about the
  Gold plan?"* are understood using recent chat history
- **Source transparency** — every answer comes with an expandable panel
  showing exactly which document chunks were used to generate it
- **Structure-aware chunking** — documents are split and grouped by
  section so related information (e.g. all four plan tiers) stays together
  in one retrievable chunk

<img width="1600" height="722" alt="WhatsApp Image 2026-06-29 at 8 07 18 PM" src="https://github.com/user-attachments/assets/c4ed0a0c-d059-4d7d-b6d2-bf02e473c297" />


## How it works

```
sample_docs/  →  chunk + embed  →  Chroma vector store
                                          │
user question  →  hybrid search (semantic + BM25)  →  rerank  →  top chunks
                                          │
                          chunks + chat history → Gemini → answer
```

## Tech stack

| Component         | Tool                                      |
|--------------------|--------------------------------------------|
| Vector store        | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| Embeddings           | `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Keyword search       | [rank-bm25](https://pypi.org/project/rank-bm25/) |
| Reranking             | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM                    | Gemini API (`gemini-2.5-flash`) |
| UI                      | Streamlit |

## Project structure

```
insurance_rag_chatbot/
├── sample_docs/          # fictional insurance documents (knowledge base)
├── ingest.py             # chunks + embeds documents into Chroma
├── app.py                # Streamlit chat app (retrieval + generation)
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   and add it to a `.env` file in the project root:

   ```
   GEMINI_API_KEY=your-key-here
   ```

3. Build the vector index (run once, or again any time you edit `sample_docs/`):

   ```
   python ingest.py
   ```

4. Launch the chatbot:

   ```
   streamlit run app.py
   ```

## Try asking

- "What are the insurance plans?"
- "What's the difference between the Silver and Gold plans?"
- "How do I file a claim for an out-of-network visit?"
- "What about dental?" *(as a follow-up — tests conversation memory)*
- "What's a deductible?"

<img width="1600" height="726" alt="Conext image" src="https://github.com/user-attachments/assets/cc89f171-a303-4b86-bc99-ab90b909b040" />


## Known limitations

This is a proof of concept, not a production system — a few deliberate
shortcuts worth knowing about if you build on this:

- **Chunk size (1500 chars) was tuned by hand** against this specific
  document set, not derived from any principled rule. A larger or
  differently-structured document set would need re-tuning.
- **No automated retrieval evaluation.** Changes were validated against
  individual test questions rather than a golden test set with
  recall/precision metrics — the right next step before trusting any
  future retrieval change.
- **BM25 tokenizer uses naive plural-stripping**, not a real stemmer —
  it can mishandle some words ending in "s" that aren't actually plurals.
- **In-memory chat history**, not persisted between sessions or restarts.
- Don't use this with real member or policy data — it's built entirely on
  fictional sample documents for demonstration purposes.

## Future improvements

- Replace size-based chunk merging with [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
  (prepending an LLM-generated "where this chunk fits" summary before indexing)
- Build a small golden test set to measure retrieval quality objectively
- Swap the BM25 tokenizer for a real stemmer (e.g. NLTK's Porter stemmer)
- Persist conversation history across sessions
- Add metadata filtering (e.g. by plan tier) to narrow retrieval further
