# backend/rag/retriever.py

import numpy as np
from .embeddings import embed_query


MIN_SCORE = 0.3


def retrieve(query, index, chunks, k=5, return_scores=False):
    if index is None or len(chunks) == 0:
        return [] if not return_scores else ([], [])

    query_embedding = embed_query(query).astype(np.float32)

    k = min(k * 2, len(chunks))  # 🔥 retrieve more for filtering

    distances, indices = index.search(query_embedding, k)

    candidates = []

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        if score < MIN_SCORE:
            continue

        candidates.append((chunks[idx], float(score)))

    # -------- Deduplicate similar chunks --------
    seen = set()
    filtered = []

    for text, score in sorted(candidates, key=lambda x: x[1], reverse=True):
        key = text[:100]  # simple fingerprint

        if key in seen:
            continue

        seen.add(key)
        filtered.append((text, score))

        if len(filtered) >= k // 2:
            break

    results = [c[0] for c in filtered]
    scores = [c[1] for c in filtered]

    if return_scores:
        return results, scores

    return results