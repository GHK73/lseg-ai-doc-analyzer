# backend/rag/chunker.py

import re


def chunk_text(text: str, chunk_size=500, overlap=50):
    # -------- Clean text --------
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # -------- Sentence split (robust fallback) --------
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # fallback if no proper sentence split
    if len(sentences) <= 1:
        sentences = text.split(" ")

    chunks = []
    current_words = []

    for sentence in sentences:
        words = sentence.split()

        # if fallback case → words already list
        if isinstance(sentence, list):
            words = sentence

        current_words.extend(words)

        if len(current_words) >= chunk_size:
            chunks.append(" ".join(current_words[:chunk_size]))

            # -------- overlap --------
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words.copy()

    # -------- remaining --------
    if current_words:
        chunks.append(" ".join(current_words))

    # -------- filter small chunks --------
    chunks = [c for c in chunks if len(c.split()) > 20]

    return chunks