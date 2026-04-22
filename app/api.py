# File: app/api.py
"""FastAPI application — exposes /ingest and /query endpoints for DocMind."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingest import ingest_file
from app.chain import ask_question

app = FastAPI(
    title="DocMind",
    description="Document Q&A agent powered by LangChain RAG, ChromaDB, and Groq.",
    version="1.0.0",
)

# Allow all origins for local development and Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ---

class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""
    file_path: str


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""
    status: str
    chunks_added: int


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str
    session_id: str = None  # optional — generated automatically if not provided


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""
    answer: str
    source_chunks: list
    relevant_history: list
    session_id: str  # always returned so client can pass it back next time


# --- Endpoints ---

@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    """Ingest a PDF or text file into the ChromaDB knowledge base."""
    try:
        chunks_added = ingest_file(request.file_path)
        return IngestResponse(status="success", chunks_added=chunks_added)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Ask a question and get an answer grounded in ingested documents and past conversation."""
    try:
        result = ask_question(
            question=request.question,
            session_id=request.session_id,
        )
        return QueryResponse(
            answer=result["answer"],
            source_chunks=result["source_chunks"],
            relevant_history=result["relevant_history"],
            session_id=result["session_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
