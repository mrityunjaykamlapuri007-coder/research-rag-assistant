# ⚗️ Research RAG Assistant

> **Production-grade Retrieval-Augmented Generation system for scientific research papers on piezoelectric ceramics.**

Built by **Mrityunjay Kumar**

---

## 🚀 Live Demo

> Ask questions across 21 research papers on CaBi₂Nb₂O₉ piezoelectric ceramics and get precise, cited answers powered by LLaMA 3.1 8B.

---

## 📌 Overview

This project implements an advanced RAG pipeline that allows users to query a corpus of scientific research papers using natural language. The system retrieves the most relevant content using hybrid search, reranks it with a cross-encoder, compresses the context intelligently, and generates accurate answers using a locally running quantized LLM.

---

## 🏗️ Architecture

```
PDFs (21 Research Papers)
         ↓
   ingestion.py
   ├── PyMuPDF block-level text extraction
   ├── Table extraction via pandas
   ├── Garbled text filtering
   ├── Smart chunking (1024 tokens / 200 overlap)
   └── BGE-Large-en-v1.5 embeddings → FAISS + BM25
         ↓
   retriever.py
   ├── Query expansion (40+ domain-specific mappings)
   ├── Multi-query generation
   ├── Hybrid Search (FAISS dense + BM25 sparse)
   ├── Section boosting (abstract +0.20, result +0.15)
   └── RRF Fusion (k=60)
         ↓
   reranker.py
   └── BGE-Reranker-Large CrossEncoder
         ↓
   context_compressor.py
   ├── Sentence-level relevance scoring
   ├── Value/keyword weighted selection
   └── NO_COMPRESS_TERMS bypass for sensitive queries
         ↓
   rag_chain.py
   ├── Query rewriting for ambiguous references
   ├── LLaMA 3.1 8B Instruct (4-bit NF4 quantized)
   ├── Strict scientific answer prompt
   └── Multi-turn chat history (last 6 turns)
         ↓
   main.py (FastAPI Backend)
   ├── POST /ask
   ├── POST /retrieve
   └── GET /health
         ↓
   app.py (Streamlit Frontend)
   ├── Custom chat UI with source citations
   ├── Retrieved chunk viewer with rerank scores
   └── Evaluation metrics dashboard
```

---

## 🔥 Key Features

| Feature | Description |
|---|---|
| **Hybrid Search** | Combines BM25 keyword search + FAISS semantic search |
| **RRF Fusion** | Reciprocal Rank Fusion merges both result sets intelligently |
| **Cross-Encoder Reranking** | BGE-Reranker-Large rescores retrieved chunks for precision |
| **Context Compression** | Sentence-level scoring reduces noise before LLM inference |
| **Query Expansion** | 40+ domain-specific term mappings (CBN↔CaBi₂Nb₂O₉, d33↔piezoelectric coefficient) |
| **Query Rewriting** | Resolves ambiguous pronouns/references using chat history |
| **4-bit Quantization** | LLaMA 3.1 8B runs locally via NF4 quantization with bitsandbytes |
| **Multi-turn Chat** | Maintains last 6 conversation turns for contextual follow-ups |
| **Section Boosting** | Abstract and results sections weighted higher in retrieval |
| **Table Extraction** | Research paper tables extracted and indexed as structured text |

---

## 📊 Evaluation Results

Evaluated on **38 domain-specific question-answer pairs** using multiple metrics:

| Metric | Start | Final | Improvement |
|---|---|---|---|
| ROUGE-1 | 0.052 | **0.284** | +446% |
| ROUGE-2 | 0.010 | **0.137** | +1270% |
| ROUGE-L | 0.039 | **0.226** | +479% |
| BERTScore F1 | 0.394 | **0.611** | +55% |
| Semantic Similarity | 0.671 | **0.840** | +25% |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| **LLM** | LLaMA 3.1 8B Instruct (4-bit NF4 via bitsandbytes) |
| **Embeddings** | BAAI/bge-large-en-v1.5 |
| **Reranker** | BAAI/bge-reranker-large (CrossEncoder) |
| **Vector Store** | FAISS (local) |
| **Sparse Search** | BM25 (rank-bm25) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Framework** | LangChain |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Evaluation** | ROUGE, BERTScore, Semantic Similarity |

---

## 📁 Project Structure

```
research-rag-assistant/
├── data/
│   └── papers/              ← 21 research PDFs
├── vectorstore/
│   ├── faiss_index/         ← FAISS vector index
│   └── chunks.json          ← BM25 chunk metadata
├── src/
│   ├── ingestion.py         ← PDF processing pipeline
│   ├── retriever.py         ← Hybrid search + RRF fusion
│   ├── reranker.py          ← CrossEncoder reranking
│   ├── context_compressor.py← Sentence-level compression
│   ├── rag_chain.py         ← LLM + prompt + chat history
│   ├── evaluator.py         ← ROUGE + BERTScore + Sem Sim
│   └── main.py              ← FastAPI backend
├── app.py                   ← Streamlit frontend
├── download_model.py        ← HuggingFace model downloader
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/mrityunjaykamlapuri007-coder/research-rag-assistant.git
cd research-rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
```bash
cp .env.example .env
```
Add your HuggingFace token to `.env`:
```
HF_TOKEN=your_huggingface_token_here
```

### 5. Download LLaMA 3.1 8B model
```bash
python download_model.py
```

### 6. Add research papers
Place your PDF files in `data/papers/`

### 7. Run ingestion pipeline
```bash
python src/ingestion.py
```

### 8. Start FastAPI backend
```bash
python src/main.py
```

### 9. Start Streamlit frontend
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser ✅

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status |
| `GET` | `/health` | Health check with model info |
| `POST` | `/ask` | Ask a question, get answer + chat history |
| `POST` | `/retrieve` | Retrieve relevant chunks for a query |

### Example request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Curie temperature of CBN ceramics?", "chat_history": []}'
```

### Example response
```json
{
  "answer": "The Curie temperature of pure CaBi₂Nb₂O₉ (CBN) ceramics is approximately 930°C, which is the highest known Tc among two-layer Aurivillius phase ceramics.",
  "chat_history": ["User: What is the Curie temperature...", "Assistant: The Curie temperature..."]
}
```

---

## 📝 Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace access token for LLaMA model |
| `API_URL` | FastAPI backend URL (default: http://localhost:8000) |

---

## 🧪 Running Evaluation

```bash
python src/evaluator.py
```

Results saved to `evaluation_results.csv`

---

## 📖 Research Domain

This system is specialized for research papers on:
- **CaBi₂Nb₂O₉ (CBN)** piezoelectric ceramics
- **Aurivillius-type** bismuth layered structure ferroelectrics
- High-temperature piezoelectric applications
- Chemical doping and substitution effects

---

## 🤝 Author

**Mrityunjay Kumar**
- GitHub: [@mrityunjaykamlapuri007-coder](https://github.com/mrityunjaykamlapuri007-coder)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## ⭐ If you found this useful, please star the repository!