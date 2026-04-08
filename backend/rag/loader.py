# backend/rag/loader.py

import fitz  # PyMuPDF
import os
import re


def _clean_text(text: str) -> str:
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # remove null characters
    text = text.replace("\x00", " ")

    return text.strip()


def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = []

    try:
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                raise ValueError("Empty PDF")

            for page_num, page in enumerate(doc):
                try:
                    # -------- Primary method: blocks --------
                    blocks = page.get_text("blocks")

                    if blocks:
                        # sort blocks top → bottom
                        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

                        page_text = " ".join(
                            block[4] for block in blocks if block[4].strip()
                        )
                    else:
                        # -------- Fallback method --------
                        page_text = page.get_text("text")

                    if page_text and page_text.strip():
                        pages.append(_clean_text(page_text))

                except Exception:
                    # skip corrupted page instead of crashing
                    continue

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    # -------- Combine all pages --------
    full_text = "\n".join(pages)

    # -------- Final validation --------
    if not full_text.strip():
        raise ValueError("No readable text found in PDF")

    return _clean_text(full_text)