from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import os
import json

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

app = FastAPI()


# -------- Helpers --------
def get_user_path(user_id: str):
    return f"data/{user_id}"


def limit_context(chunks, max_chars=3000):
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n"
    return context


# -------- Routes --------
@app.get("/")
def home():
    return {"message": "RAG System Running"}


# -------- Upload --------
@app.post("/upload")
def upload_file(user_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    user_path = get_user_path(user_id)
    os.makedirs(user_path, exist_ok=True)

    file_path = os.path.join(user_path, os.path.basename(file.filename))

    # ---- Save file ----
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---- Load existing store ----
    index, _ = load_vector_store(user_path)
    chunks_path = os.path.join(user_path, "chunks.json")

    if os.path.exists(chunks_path):
        with open(chunks_path) as f:
            chunks = json.load(f)
    else:
        chunks = []

    # ---- Process PDF ----
    text = load_pdf(file_path)
    new_chunks = chunk_text(text)

    # ---- Create or update index ----
    if index is None:
        index, embeddings = create_vector_store(new_chunks)
    else:
        add_to_index(index, new_chunks)

    # ---- Update chunks ----
    chunks.extend(new_chunks)

    # ---- Save ----
    save_vector_store(index, path=user_path)

    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

    return {
        "message": "File uploaded and indexed",
        "chunks_added": len(new_chunks)
    }


# -------- Query --------
@app.get("/ask")
def ask(query: str, user_id: str):
    user_path = get_user_path(user_id)

    index, _ = load_vector_store(user_path)
    chunks_path = os.path.join(user_path, "chunks.json")

    if index is None or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="No data found for this user")

    with open(chunks_path) as f:
        chunks = json.load(f)

    retrieved_chunks = retrieve(query, index, chunks, k=5)
    context = limit_context(retrieved_chunks)

    answer = generate_answer(query, context)

    return {"answer": answer}