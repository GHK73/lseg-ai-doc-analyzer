# backend/rag/retriever.py

import numpy as np
from .embeddings import embed_query


MIN_SCORE = 0.3  # 🔴 tune this (0.2–0.4 depending on data)


def retrieve(query, index, chunks, k=5, return_scores=False):
    if index is None or len(chunks) == 0:
        return [] if not return_scores else ([], [])

    # ---- Embed query ----
    query_embedding = embed_query(query).astype(np.float32)

    # ---- Ensure k valid ----
    k = min(k, len(chunks))

    # ---- FAISS search ----
    distances, indices = index.search(query_embedding, k)

    candidates = []

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        if score < MIN_SCORE:
            continue  # 🔴 filter weak matches

        candidates.append((chunks[idx], float(score)))

    # 🔴 enforce deterministic ordering
    candidates.sort(key=lambda x: x[1], reverse=True)

    results = [c[0] for c in candidates]
    scores = [c[1] for c in candidates]

    if return_scores:
        return results, scores

    return results