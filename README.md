# ⚗️ Research RAG Assistant

<p align="center">
  <img src="https://img.shields.io/badge/LLM-LLaMA%203.1%208B-blue?style=for-the-badge&logo=meta" />
  <img src="https://img.shields.io/badge/Embeddings-BGE--Large-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Search-FAISS%20%2B%20BM25-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-teal?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> Production-grade Retrieval-Augmented Generation system for scientific research papers on piezoelectric ceramics. Ask natural language questions across 21 research papers and get precise, cited answers powered by LLaMA 3.1 8B.

**Built by [Mrityunjay Kumar](https://github.com/mrityunjaykamlapuri007-coder)**

---

## 🎬 Demo

<!-- Add demo GIF here after recording -->
> _Demo GIF coming soon — Streamlit UI with source citations and rerank scores_

---

## 📊 Evaluation Results

Evaluated on **42 domain-specific QA pairs** across the research corpus:

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| ROUGE-1 | 0.052 | 0.284 | **+446%** |
| ROUGE-2 | 0.010 | 0.137 | **+1270%** |
| ROUGE-L | 0.039 | 0.226 | **+479%** |
| BERTScore F1 | 0.394 | 0.611 | **+55%** |
| Semantic Similarity | 0.671 | 0.840 | **+25%** |

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
|---------|-------------|
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

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | LLaMA 3.1 8B Instruct (4-bit NF4 via bitsandbytes) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Reranker | BAAI/bge-reranker-large (CrossEncoder) |
| Vector Store | FAISS (local) |
| Sparse Search | BM25 (rank-bm25) |
| PDF Parsing | PyMuPDF (fitz) |
| Framework | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | ROUGE, BERTScore, Semantic Similarity |

---

## 📁 Project Structure

```
research-rag-assistant/
├── data/
│   └── papers/               ← 21 research PDFs
├── src/
│   ├── ingestion.py          ← PDF processing pipeline
│   ├── retriever.py          ← Hybrid search + RRF fusion
│   ├── reranker.py           ← CrossEncoder reranking
│   ├── context_compressor.py ← Sentence-level compression
│   ├── rag_chain.py          ← LLM + prompt + chat history
│   ├── evaluator.py          ← ROUGE + BERTScore + Sem Sim
│   └── main.py               ← FastAPI backend
├── kaggle/
│   └── kaggle_chain_and_evaluation.ipynb  ← Kaggle notebook (GPU)
├── app.py                    ← Streamlit frontend
├── download_model.py         ← HuggingFace model downloader
├── requirements.txt
├── .env.example
└── README.md
```

> **Note:** `vectorstore/` and `models/` are excluded from git (see `.gitignore`). Generate them locally using the setup steps below.

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

### 6. Run ingestion pipeline
```bash
python src/ingestion.py
```

### 7. Start FastAPI backend
```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 8. Start Streamlit frontend
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) ✅

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with model info |
| POST | `/ask` | Ask a question, get answer + sources |
| POST | `/retrieve` | Retrieve relevant chunks for a query |

**Example request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Curie temperature of CBN ceramics?", "chat_history": []}'
```

**Example response:**
```json
{
  "answer": "The Curie temperature of pure CaBi₂Nb₂O₉ (CBN) ceramics is approximately 930°C, the highest known Tc among two-layer Aurivillius phase ceramics.",
  "sources": [
    {"source": "Liu 2018", "page": 3, "section": "result"}
  ]
}
```

---

## 🧪 Running Evaluation

```bash
python src/evaluator.py
```

Results saved to `evaluation_results.csv`. Full evaluation notebook available in `kaggle/`.

---

## 📖 Research Domain

This system is specialized for research papers on:
- CaBi₂Nb₂O₉ (CBN) piezoelectric ceramics
- Aurivillius-type bismuth layered structure ferroelectrics (BLSF)
- High-temperature piezoelectric applications
- Chemical doping and A-site/B-site substitution effects

---

## 🤝 Author

**Mrityunjay Kumar**
GitHub: [@mrityunjaykamlapuri007-coder](https://github.com/mrityunjaykamlapuri007-coder)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this useful, please star the repository!