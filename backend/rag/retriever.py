# backend/rag/retriver.py

import numpy as np
from .embeddings import embed_query


def retrieve(query, index, chunks, k=3, return_scores=False):
    if index is None or len(chunks) == 0:
        return [] if not return_scores else ([], [])

    # ---- Embed query ----
    query_embedding = embed_query(query).astype(np.float32)

    # ---- Ensure k is valid ----
    k = min(k, len(chunks))

    # ---- FAISS search ----
    distances, indices = index.search(query_embedding, k)

    results = []
    scores = []

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        results.append(chunks[idx])
        scores.append(float(score))

    if return_scores:
        return results, scores

    return results