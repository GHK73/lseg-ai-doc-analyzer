# backend/rag/embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def create_vector_store(chunks, batch_size=32):
    model = get_model()

    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings