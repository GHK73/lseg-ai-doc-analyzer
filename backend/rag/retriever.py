import numpy as np
from .embeddings import embed_query


def retrieve(query, index, chunks, k=5):
    """
    Retrieve top-k relevant chunks.

    Args:
        query (str)
        index (FAISS index)
        chunks (list): 
            [
                "text"
                OR
                {"text": "...", "page": 1}
            ]
        k (int)

    Returns:
        List[dict]:
        [
            {"text": "...", "page": 2, "score": 0.91}
        ]
    """

    if index is None or not chunks:
        return []

    # -------- Embed query --------
    query_embedding = embed_query(query).astype(np.float32)

    # -------- FAISS search --------
    search_k = min(k * 4, len(chunks))  # over-fetch
    scores, indices = index.search(query_embedding, search_k)

    candidates = []

    # -------- Collect candidates --------
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            page = chunk.get("page", "N/A")
        else:
            text = chunk
            page = "N/A"

        if not text:
            continue

        candidates.append({
            "text": text,
            "page": page,
            "score": float(score)
        })

    if not candidates:
        return []

    # -------- Sort by relevance --------
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # -------- Deduplicate --------
    seen = set()
    results = []

    for item in candidates:
        key = item["text"][:150]  # fingerprint

        if key in seen:
            continue

        seen.add(key)
        results.append(item)

        if len(results) >= k:
            break

    # -------- Debug --------
    print("🔍 Top scores:", [round(s, 3) for s in scores[0][:5]])

    return results