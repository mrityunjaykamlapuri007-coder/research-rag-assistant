from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from rag_chain import get_answer
from retriever import retrieve

app = FastAPI(
    title="Research RAG API",
    description="Question Answering API for scientific research papers on piezoelectric ceramics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[str]] = []


class QuestionResponse(BaseModel):
    answer: str
    chat_history: List[str]


class ChunkResponse(BaseModel):
    text: str
    source: str
    page: str
    section: str
    rerank_score: Optional[float] = None


class RetrieveResponse(BaseModel):
    question: str
    chunks: List[ChunkResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    papers: str
    embedding_model: str


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Research RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok",
        model="Llama 3.1 8B (4-bit NF4)",
        papers="21 research papers on CBN piezoelectric ceramics",
        embedding_model="BAAI/bge-large-en-v1.5"
    )


@app.post("/ask", response_model=QuestionResponse, tags=["QA"])
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer, updated_history = get_answer(
            request.question,
            request.chat_history or []
        )
        return QuestionResponse(
            answer=answer,
            chat_history=updated_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
def retrieve_chunks(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        chunks = retrieve(request.question)
        return RetrieveResponse(
            question=request.question,
            chunks=[
                ChunkResponse(
                    text=c["text"],
                    source=c.get("source", "Unknown"),
                    page=str(c.get("page", "?")),
                    section=c.get("section", "?"),
                    rerank_score=c.get("rerank_score", None)
                )
                for c in chunks
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)