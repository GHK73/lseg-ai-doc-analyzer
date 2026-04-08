from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import threading
import pickle
import torch


# -------- Thread-safe Lazy Model Loading --------
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("🚀 Loading embedding model...")

                device = "cuda" if torch.cuda.is_available() else "cpu"

                _model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device=device
                )
                _model.eval()

    return _model


# -------- Embedding Functions --------
def embed_texts(texts, batch_size=32):
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

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
    """
    chunks can be:
    - List[str]
    - List[{"text": "..."}]
    """

    if not chunks:
        raise ValueError("No chunks provided")

    texts = [
        c["text"] if isinstance(c, dict) else c
        for c in chunks
    ]

    embeddings = embed_texts(texts, batch_size=batch_size)

    dimension = embeddings.shape[1]

    # faster scalable index
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efSearch = 64

    index.add(embeddings)

    return index, embeddings


def add_to_index(index, new_chunks):
    if not new_chunks:
        return None

    texts = [
        c["text"] if isinstance(c, dict) else c
        for c in new_chunks
    ]

    new_embeddings = embed_texts(texts)
    index.add(new_embeddings)

    return new_embeddings


# -------- Persistence --------
def save_vector_store(index, embeddings=None, chunks=None, path="data"):
    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, os.path.join(path, "faiss.index"))

    if embeddings is not None:
        np.save(os.path.join(path, "embeddings.npy"), embeddings)

    if chunks is not None:
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)


def load_vector_store(path="data"):
    index_path = os.path.join(path, "faiss.index")
    emb_path = os.path.join(path, "embeddings.npy")
    chunk_path = os.path.join(path, "chunks.pkl")

    if not os.path.exists(index_path):
        return None, None, None

    index = faiss.read_index(index_path)

    embeddings = None
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)

        if index.ntotal != embeddings.shape[0]:
            raise ValueError("FAISS index and embeddings mismatch")

    chunks = None
    if os.path.exists(chunk_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)

        if embeddings is not None and len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings mismatch")

    return index, embeddings, chunks