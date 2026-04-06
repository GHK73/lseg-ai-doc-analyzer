# backend/rag/embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(chunks):
    embeddings = model.encode(chunks)
    embeddings = embeddings / np.linlag.norm(embeddings, axis = 1,keepdims = True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index,embeddings