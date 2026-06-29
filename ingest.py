
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "sample_docs"
DB_DIR = "chroma_db"
COLLECTION_NAME = "insurance_docs"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks


def load_documents():
    """Read every .md file in DOCS_DIR."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text, chunk_size=800, overlap=150):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += "\n\n" + p
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def main():
    print("Loading sample insurance documents...")
    documents = load_documents()
    print(f"Found {len(documents)} documents")

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

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
    embeddings = model.encode(texts).tolist()

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Done. Stored {len(texts)} chunks in '{DB_DIR}/'.")


if __name__ == "__main__":
    main()
