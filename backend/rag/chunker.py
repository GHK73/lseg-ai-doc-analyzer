import re


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30):
    """
    Split text into overlapping chunks.

    Args:
        text (str): input text
        chunk_size (int): words per chunk
        overlap (int): overlap between chunks

    Returns:
        List[str]: list of chunks
    """

    # -------- Clean text --------
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # -------- Sentence splitting --------
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+',
        text
    )

    if len(sentences) <= 1:
        sentences = [text]  # fallback

    # -------- Chunking --------
    chunks = []
    buffer = []

    for sentence in sentences:
        words = sentence.split()
        buffer.extend(words)

        while len(buffer) >= chunk_size:
            chunk_words = buffer[:chunk_size]
            chunks.append(" ".join(chunk_words))

            # apply overlap
            buffer = buffer[chunk_size - overlap:] if overlap > 0 else []

    # -------- Remaining --------
    if buffer:
        chunks.append(" ".join(buffer))

    # -------- Filter tiny chunks --------
    return [
        chunk for chunk in chunks
        if len(chunk.split()) > 30
    ]