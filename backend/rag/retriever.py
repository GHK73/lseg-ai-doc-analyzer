# backend/rag/retriver.py

import numpy as np
from .embeddings import embed_query


def retrieve(query, index, chunks, k=3, return_scores=False):
    # ---- Embed query (consistent with doc embeddings) ----
    query_embedding = embed_query(query)

    # ---- FAISS search ----
    distances, indices = index.search(query_embedding, k)

    # ---- Safe retrieval ----
    results = []
    scores = []

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append(chunks[idx])
        scores.append(float(score))

    if return_scores:
        return results, scores

    return results
