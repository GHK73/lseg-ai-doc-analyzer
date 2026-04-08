# backend/rag/embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import threading

# -------- Thread-safe Lazy Model Loading --------
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("🚀 Loading embedding model...")
                _model = SentenceTransformer("all-MiniLM-L6-v2")
                _model.eval()
    return _model


# -------- Embedding Functions --------
def embed_texts(texts, batch_size=32):
    if not texts:
        return np.array([], dtype=np.float32)

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # safety check
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")

    return embeddings


def embed_query(query: str):
    if not query.strip():
        raise ValueError("Query cannot be empty")

    model = get_model()

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim != 2:
        raise ValueError("Query embedding must be 2D")

    return embedding


# -------- FAISS Vector Store --------
def create_vector_store(chunks, batch_size=32):
    if not chunks:
        raise ValueError("No chunks provided")

    embeddings = embed_texts(chunks, batch_size=batch_size)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity

    index.add(embeddings)

    return index, embeddings


def add_to_index(index, new_chunks):
    if not new_chunks:
        return None

    new_embeddings = embed_texts(new_chunks)
    index.add(new_embeddings)

    return new_embeddings


# -------- Persistence --------
def save_vector_store(index, embeddings=None, path="data"):
    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, os.path.join(path, "faiss.index"))

    if embeddings is not None:
        np.save(os.path.join(path, "embeddings.npy"), embeddings)


def load_vector_store(path="data"):
    index_path = os.path.join(path, "faiss.index")
    emb_path = os.path.join(path, "embeddings.npy")

    if not os.path.exists(index_path):
        return None, None

    index = faiss.read_index(index_path)

    embeddings = None
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)

        # strict consistency check
        if index.ntotal != embeddings.shape[0]:
            raise ValueError("FAISS index and embeddings mismatch")

    return index, embeddings