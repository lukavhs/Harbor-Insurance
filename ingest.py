"""
Ingest sample insurance documents into a local Chroma vector store.

Run this once before starting the chatbot, and again any time you edit
the files in sample_docs/:

    python ingest.py
"""

import os
import re
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "sample_docs"
DB_DIR = "chroma_db"
COLLECTION_NAME = "insurance_docs"
CHUNK_SIZE = 1500      # characters per chunk
CHUNK_OVERLAP = 200    # carried-forward overlap, used only when a section
                       # is too big to fit in one chunk on its own

# bge-small is trained specifically for query -> passage retrieval (asymmetric
# search), unlike general-purpose sentence-similarity models. Documents are
# embedded as-is here; queries get a special instruction prefix at query time
# in app.py - that asymmetry is part of how the model was trained and is what
# improves "does this chunk actually answer the question" matching.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_documents():
    """Read every .md file in DOCS_DIR."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Group related sections together instead of one-chunk-per-header.

    Header-delimited sections are greedily packed into chunks up to
    chunk_size, so e.g. all four plan tiers in a short overview doc end up
    in ONE chunk together - much easier to retrieve for "what are the
    plans?" - instead of four isolated chunks that don't even contain the
    word "plans" anywhere in their body.

    Any single section that's still too big on its own falls back to
    paragraph-level packing (_split_oversized), so we never cut content
    mid-sentence or mid-entry.
    """
    sections = re.split(r'\n(?=#{1,3} )', text)
    sections = [s.strip() for s in sections if s.strip()]

    raw_chunks = []
    current = ""
    for section in sections:
        if not current:
            current = section
        elif len(current) + 2 + len(section) <= chunk_size:
            current = current + "\n\n" + section
        else:
            raw_chunks.append(current)
            current = section
    if current:
        raw_chunks.append(current)

    final_chunks = []
    for chunk in raw_chunks:
        final_chunks.extend(_split_oversized(chunk, chunk_size, overlap))
    return final_chunks


def _split_oversized(text, chunk_size, overlap):
    """Fallback for a single section too big to fit in one chunk on its
    own (e.g. the FAQ and glossary docs, which have no ## subheaders to
    split on). Packs whole paragraphs - never mid-sentence - and carries
    the last paragraph forward into the next piece for a bit of overlap.
    """
    if len(text) <= chunk_size:
        return [text]

    header_match = re.match(r'(#{1,3} .+)', text)
    header = header_match.group(1) if header_match else ""
    body = text[len(header):].strip() if header else text
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    pieces = []
    current_paragraphs = []
    for p in paragraphs:
        candidate_paragraphs = current_paragraphs + [p]
        candidate_body = "\n\n".join(candidate_paragraphs)
        candidate = header + "\n\n" + candidate_body if header else candidate_body
        if len(candidate) <= chunk_size or not current_paragraphs:
            current_paragraphs = candidate_paragraphs
        else:
            body_text = "\n\n".join(current_paragraphs)
            pieces.append(header + "\n\n" + body_text if header else body_text)
            carry_forward = current_paragraphs[-1] if overlap > 0 else None
            current_paragraphs = [carry_forward, p] if carry_forward else [p]
    if current_paragraphs:
        body_text = "\n\n".join(current_paragraphs)
        pieces.append(header + "\n\n" + body_text if header else body_text)
    return pieces


def main():
    print("Loading sample insurance documents...")
    documents = load_documents()
    print(f"Found {len(documents)} documents")

    print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)

    # Start fresh each time this script runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['source']}-{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})

    print(f"Created {len(texts)} chunks. Embedding...")
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Done. Stored {len(texts)} chunks in '{DB_DIR}/'.")


if __name__ == "__main__":
    main()