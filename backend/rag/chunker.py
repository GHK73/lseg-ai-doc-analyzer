# backend/rag/chunker.py

import re


def chunk_text(text, chunk_size=500, overlap=50):
    # ---- Clean text ----
    text = re.sub(r"\s+", " ", text).strip()

    # ---- Sentence split ----
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)

        # compute current length (words, not chars)
        words = " ".join(current_chunk).split()

        if len(words) >= chunk_size:
            chunks.append(" ".join(words))

            # ---- FIXED overlap (word-based) ----
            overlap_words = words[-overlap:] if overlap > 0 else []
            current_chunk = [" ".join(overlap_words)]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # ---- filter tiny chunks ----
    chunks = [c for c in chunks if len(c.split()) > 20]

    return chunks