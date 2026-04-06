# backend/app.py

from fastapi import FastAPI
from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import create_vector_store,model
from rag.retriever import retrieve
from rag.qa import generate_answer

app = FastAPI()

text = load_pdf("../data/sample.pdf")
chunks = chunk_text(text)
index,embeddings = create_vector_store(chunks)

@app.get("/") 
def home():
    return {"message": "RAG System Running"}


@app.get("/ask")
def ask(query: str):
    retrieved_chunks = retrieve(query, model, index, chunks, k=5)
    context = "\n".join(retrieved_chunks)

    answer = generate_answer(query, context)

    return {"answer": answer}