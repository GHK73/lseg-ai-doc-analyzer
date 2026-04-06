# backend/rag/loader.py

import fitz
import os


def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = []

    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text.append(page_text.strip())

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    return "\n".join(text)