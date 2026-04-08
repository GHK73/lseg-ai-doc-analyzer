# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from document.routes import router as document_router
from auth.routes import router as auth_router


app = FastAPI()
app.include_router(auth_router, prefix="/auth")


# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Base Route --------
@app.get("/")
def home():
    return {"message": "RAG System Running"}

# -------- Include Routes --------
app.include_router(document_router, prefix="/doc")