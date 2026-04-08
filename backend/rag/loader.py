import fitz  # PyMuPDF
import os
import re


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", " ")
    return text.strip()


def _extract_page_text(page):
    """
    Extract text from a page using best available method.
    """

    # -------- Try block extraction (better layout) --------
    blocks = page.get_text("blocks")

    if blocks:
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        text = " ".join(
            block[4] for block in blocks if block[4].strip()
        )
    else:
        # fallback
        text = page.get_text("text")

    return _clean_text(text)


def load_pdf(file_path: str):
    """
    Returns:
    [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."}
    ]
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = []

    try:
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                raise ValueError("Empty PDF")

            for page_num, page in enumerate(doc):
                try:
                    page_text = _extract_page_text(page)

                    if page_text:
                        pages.append({
                            "page": page_num + 1,
                            "text": page_text
                        })

                except Exception:
                    # skip bad pages
                    continue

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    if not pages:
        raise ValueError("No readable text found in PDF")

    return pages