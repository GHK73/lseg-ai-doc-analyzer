# backend/rag/embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def save_vector_store(index, embeddings, path="data"):
    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, os.path.join(path, "faiss.index"))
    np.save(os.path.join(path, "embeddings.npy"), embeddings)


def load_vector_store(path="data"):
    index_path = os.path.join(path, "faiss.index")
    emb_path = os.path.join(path, "embeddings.npy")

    if not os.path.exists(index_path) or not os.path.exists(emb_path):
        return None, None

    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)

    return index, embeddings



def create_vector_store(chunks, batch_size=32):
    model = get_model()

    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, a_min=1e-10, a_max=None)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings