# backend/rag/retriever.py

import numpy as np
from .embeddings import embed_query


def retrieve(query, index, chunks, k=5, return_scores=False):
    """
    Retrieve top-k relevant chunks using FAISS similarity search.
    """

    if index is None or len(chunks) == 0:
        return ([], []) if return_scores else []

    # -------- Embed query --------
    query_embedding = embed_query(query).astype(np.float32)

    # -------- Search FAISS --------
    search_k = min(k * 3, len(chunks))  # over-fetch for better filtering
    scores, indices = index.search(query_embedding, search_k)

    # -------- Collect candidates --------
    candidates = []

    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        candidates.append((chunks[idx], float(score)))

    if not candidates:
        return ([], []) if return_scores else []

    # -------- Sort by score --------
    candidates.sort(key=lambda x: x[1], reverse=True)

    # -------- Deduplicate --------
    seen = set()
    filtered = []

    for text, score in candidates:
        key = text[:120]  # better fingerprint

        if key in seen:
            continue

        seen.add(key)
        filtered.append((text, score))

        if len(filtered) >= k:
            break

    # -------- Final output --------
    results = [item[0] for item in filtered]
    final_scores = [item[1] for item in filtered]

    # -------- Debug (optional) --------
    print("🔍 Top scores:", scores[0][:5])

    if return_scores:
        return results, final_scores

    return results