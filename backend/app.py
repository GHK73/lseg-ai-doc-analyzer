# backend/app.py

from fastapi import FastAPI, HTTPException
from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import create_vector_store, load_vector_store, save_vector_store
from rag.retriever import retrieve
from rag.qa import generate_answer

app = FastAPI()

vector_store = {}


# -------- Helpers --------
def limit_context(chunks, max_chars=3000):
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk + "\n"
    return context


# -------- Startup --------
@app.on_event("startup")
def startup_event():
    index, embeddings = load_vector_store("data")

    if index is None:
        # first run → build
        text = load_pdf("../data/sample.pdf")
        chunks = chunk_text(text)

        index, embeddings = create_vector_store(chunks)
        save_vector_store(index, embeddings, "data")

        vector_store["chunks"] = chunks
    else:
        # load existing
        text = load_pdf("../data/sample.pdf")
        chunks = chunk_text(text)

        vector_store["chunks"] = chunks

    vector_store["index"] = index

# -------- Routes --------
@app.get("/")
def home():
    return {"message": "RAG System Running"}


@app.get("/ask")
def ask(query: str):
    if "index" not in vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    index = vector_store["index"]
    chunks = vector_store["chunks"]

    retrieved_chunks = retrieve(query, index, chunks, k=5)
    context = limit_context(retrieved_chunks)

    answer = generate_answer(query, context)

    return {"answer": answer}