# backend/rag/chunker.py

import re


def chunk_text(text, chunk_size=500, overlap=50):
    # ---- Clean text ----
    text = re.sub(r"\s+", " ", text).strip()

    # ---- Sentence split ----
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []

    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # ---- overlap handling ----
            overlap_text = " ".join(current_chunk)[-overlap:]
            current_chunk = [overlap_text, sentence]
            current_length = len(overlap_text) + sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # ---- filter tiny chunks ----
    chunks = [c for c in chunks if len(c.strip()) > 50]

    return chunks