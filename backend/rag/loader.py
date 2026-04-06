# backend/rag/loader.py

import fitz  # PyMuPDF
import os
import re


def _clean_text(text: str) -> str:
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # remove weird unicode artifacts
    text = text.replace("\x00", " ")

    return text.strip()


def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = []

    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                # 🔴 FIX: use blocks for better structure
                blocks = page.get_text("blocks")

                # sort blocks top → bottom
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

                page_text = " ".join(block[4] for block in blocks if block[4].strip())

                if page_text:
                    pages.append(_clean_text(page_text))

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    full_text = "\n".join(pages)

    # final safety cleanup
    return _clean_text(full_text)