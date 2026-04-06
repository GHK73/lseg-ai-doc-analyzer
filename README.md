# 📊 Financial Document Analyzer (RAG + ML)

Production-grade AI system for analyzing financial documents using **Retrieval-Augmented Generation (RAG)** and a modular **PyTorch-based classifier**.

---

## 🚀 Features

* 📄 PDF ingestion & parsing
* ✂️ Intelligent text chunking
* 🔎 Semantic search with FAISS
* 🧠 RAG pipeline using LLM (Groq - Llama 3.1)
* 🏷️ Document classification (PyTorch, embedding-based)
* ⚡ FastAPI backend

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

                 └──→ ML Pipeline (PyTorch Classifier)
```

---

## 📁 Project Structure

```
lseg-ai-doc-analyzer/
│
├── backend/
│   ├── app.py
│   ├── config.py   
│   │
│   ├── rag/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── qa.py
│   │
│   ├── ml/
│   │   ├── model.py        # Neural network architecture
│   │   ├── service.py      # Training + inference logic
│   │   ├── dataset.py      # Dataset handling (optional scaling)
│   │   ├── utils.py        # Save/load helpers
│   │   └── __init__.py
│   │
│   └── requirements.txt
│
├── data/
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
MODEL_NAME = "llama3-70b-8192"
```

---

## ▶️ Run Backend

```
uvicorn app:app --reload
```

---

## 🔄 RAG Pipeline

1. Load PDF → `loader.py`
2. Chunk text → `chunker.py`
3. Generate embeddings → `embeddings.py`
4. Store & retrieve via FAISS → `retriever.py`
5. Generate answer → `qa.py`

---

## 🧠 ML Pipeline (PyTorch)

### Structure

* `model.py` → defines neural network
* `service.py` → training & inference
* `dataset.py` → dataset abstraction (for scaling)
* `utils.py` → model persistence

---

### Flow

```
Embedding → Classifier → Label
```

---

### Example Usage

```
from ml.service import ClassifierService

classifier = ClassifierService(input_dim=768, num_classes=4)

# training
classifier.train(X_train, y_train)

# inference
label = classifier.predict([embedding])[0]
```

---

## 📌 Use Cases

* Financial document classification
* Semantic Q&A over reports
* Intelligent document routing
* Compliance / risk extraction

---

## ⚡ Performance Notes

* Embeddings reused across RAG + ML
* Lightweight PyTorch model (CPU-friendly)
* FAISS for fast retrieval
* Modular design for scalability

---

## 🚧 Future Improvements

* Model persistence + versioning
* Async FastAPI endpoints
* Batch inference
* Vector DB (Pinecone / Weaviate)
* Advanced classifiers (XGBoost / fine-tuning)

---

## 🧪 Tech Stack

* FastAPI
* SentenceTransformers
* FAISS
* Groq (Llama 3.1)
* PyTorch

---

## 🎯 Goal

Built to meet **production AI/ML standards (LSEG-level)**:

* modular architecture
* efficient pipelines
* scalable design

---

## 📬 Contributing

Focus areas:

* performance optimization
* ML improvements
* real-world datasets
