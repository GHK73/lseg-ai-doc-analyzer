# 📊 Financial Document Analyzer (RAG + ML)

Production-grade AI system for analyzing financial documents using **Retrieval-Augmented Generation (RAG)** and a lightweight **PyTorch classifier**.

---

## 🚀 Features

* 📄 PDF ingestion & parsing
* ✂️ Intelligent text chunking
* 🔎 Semantic search with FAISS
* 🧠 RAG pipeline using LLM (Groq - Llama 3.1)
* 🏷️ Document classification (PyTorch, embedding-based)
* ⚡ FastAPI backend (low latency APIs)

---

## 🧱 Architecture

```
PDF → Loader → Chunker → Embeddings → FAISS
                                      ↓
                               Retriever
                                      ↓
                                   LLM (Groq)
                                      ↓
                                   Answer

                 └──→ Classifier (PyTorch)
```

---

## 📁 Project Structure

```
lseg-ai-doc-analyzer/
│
├── backend/
│   ├── app.py              # FastAPI entrypoint
│   ├── config.py           # Configurations (API keys, paths)
│   │
│   ├── rag/
│   │   ├── loader.py       # PDF loading
│   │   ├── chunker.py      # Text chunking
│   │   ├── embeddings.py   # SentenceTransformers embeddings
│   │   ├── retriever.py    # FAISS search
│   │   └── qa.py           # RAG + LLM pipeline
│   │
│   ├── ml/
│   │   └── classifier.py   # PyTorch classifier (embedding-based)
│   │
│   └── requirements.txt
│
├── data/                   # PDFs + processed artifacts
└── README.md
```

---

## ⚙️ Setup

### 1. Clone repo

```
git clone <repo-url>
cd lseg-ai-doc-analyzer/backend
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 🔑 Configuration

Update `config.py`:

```
GROQ_API_KEY = "your_api_key"
MODEL_NAME = "llama3-70b-8192"  # or your choice
```

---

## ▶️ Run Backend

```
uvicorn app:app --reload
```

API will be live at:

```
http://127.0.0.1:8000
```

---

## 🔄 RAG Pipeline Flow

1. Load PDF → `loader.py`
2. Split into chunks → `chunker.py`
3. Generate embeddings → `embeddings.py`
4. Store/search via FAISS → `retriever.py`
5. Generate answer → `qa.py`

---

## 🧠 ML Classifier

* Input: embedding vector (from SentenceTransformers)
* Model: lightweight MLP (PyTorch)
* Output: document class

### Example usage

```
classifier.predict([embedding])
```

---

## 📌 Use Cases

* Financial report analysis
* Invoice classification
* Risk/compliance extraction
* Semantic Q&A over documents

---

## ⚡ Performance Notes

* Embeddings reused across RAG + ML
* CPU-only PyTorch (lightweight)
* FAISS enables fast similarity search
* Modular design for scaling

---

## 🚧 Future Improvements

* Batch processing for embeddings
* Model persistence (save/load)
* Async FastAPI endpoints
* Vector DB migration (Pinecone / Weaviate)
* Better classification (XGBoost / fine-tuned models)

---

## 🧪 Tech Stack

* FastAPI
* SentenceTransformers
* FAISS
* Groq (Llama 3.1)
* PyTorch

---

## 🎯 Goal

Designed to meet **production standards for AI/ML roles (e.g., LSEG)**:

* Modular architecture
* Efficient pipelines
* Real-world scalability

---

## 📬 Contributing

PRs and improvements welcome.
Focus on:

* performance
* modularity
* real-world use cases
