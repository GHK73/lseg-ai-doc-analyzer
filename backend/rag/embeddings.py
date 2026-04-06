# backend/rag/embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# -------- Lazy Model Loading --------
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# -------- Normalization --------
def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-10, a_max=None)


# -------- Embedding Functions --------
def embed_texts(texts, batch_size=32):
    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    return _normalize(embeddings)


def embed_query(query: str):
    model = get_model()

    embedding = model.encode([query])
    embedding = np.array(embedding, dtype=np.float32)

    return _normalize(embedding)


# -------- FAISS Vector Store --------
def create_vector_store(chunks, batch_size=32):
    embeddings = embed_texts(chunks, batch_size=batch_size)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def add_to_index(index, new_chunks):
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

    return index, embeddings