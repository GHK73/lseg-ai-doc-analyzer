# backend/document/service.py

from fastapi import HTTPException, UploadFile
import os
import json
import shutil

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    add_to_index
)
from rag.retriever import retrieve
from rag.qa import generate_answer


# -------- Helpers --------
def get_user_path(user_id: str):
    base_path = "data"
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, user_id)


def _load_chunks(chunks_path):
    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            return json.load(f)
    return []


def _save_chunks(chunks, chunks_path):
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)


# -------- Upload Service --------
def upload_document(user_id: str, file: UploadFile):
    # ---- Validate file ----
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    user_path = get_user_path(user_id)
    os.makedirs(user_path, exist_ok=True)

    file_path = os.path.join(user_path, os.path.basename(file.filename))

    # ---- Prevent duplicate uploads ----
    if os.path.exists(file_path):
        return {"message": "File already uploaded"}

    # ---- Save file ----
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    # ---- Load existing FAISS + chunks ----
    index, _ = load_vector_store(user_path)
    chunks_path = os.path.join(user_path, "chunks.json")
    chunks = _load_chunks(chunks_path)

    # ---- Extract text ----
    try:
        text = load_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ---- Chunk text ----
    new_chunks = chunk_text(text)

    if not new_chunks:
        raise HTTPException(status_code=400, detail="No valid text extracted")

    # ---- Create or update index ----
    if index is None:
        index, embeddings = create_vector_store(new_chunks)
    else:
        add_to_index(index, new_chunks)

    # ---- Update chunks ----
    chunks.extend(new_chunks)

    # ---- Save everything ----
    save_vector_store(index, path=user_path)
    _save_chunks(chunks, chunks_path)

    return {
        "message": "File uploaded and indexed",
        "chunks_added": len(new_chunks),
        "total_chunks": len(chunks)
    }


# -------- Query Service --------
def ask_question(user_id: str, query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    user_path = get_user_path(user_id)

    # ---- Load index + chunks ----
    index, _ = load_vector_store(user_path)
    chunks_path = os.path.join(user_path, "chunks.json")

    if index is None or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="No data found")

    chunks = _load_chunks(chunks_path)

    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks available")

    # ---- Retrieve relevant chunks ----
    retrieved_chunks = retrieve(query, index, chunks, k=10)

    if not retrieved_chunks:
        return {"answer": "Not found in the document"}

    # ---- Generate answer ----
    try:
        answer = generate_answer(query, retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return {
        "answer": answer,
        "chunks_used": len(retrieved_chunks)
    }