# File: app/api.py
"""FastAPI application — exposes /ingest and /query endpoints for DocMind."""

import os
from uuid import UUID

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import mcp.types as mcp_types
from pydantic import BaseModel, ValidationError
from mcp.server.sse import SseServerTransport

from app.ingest import ingest_file
from app.chain import ask_question
from app.ui import build_demo
from mcp_server.server import server as mcp_server

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
    session_id: str | None = None  # optional — generated automatically if not provided


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


# --- MCP over network (SSE transport) ---

sse_transport = SseServerTransport("/mcp/messages")


@app.get("/mcp/sse")
async def mcp_sse(request: Request):
    """Open an MCP SSE stream for remote clients."""
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options(),
        )


@app.post("/mcp/messages")
async def mcp_messages(request: Request):
    """Receive MCP client POST messages bound to an SSE session."""
    session_id_param = request.query_params.get("session_id")
    if session_id_param is None:
        return Response("session_id is required", status_code=400)

    try:
        session_id = UUID(hex=session_id_param)
    except ValueError:
        return Response("Invalid session ID", status_code=400)

    writer = sse_transport._read_stream_writers.get(session_id)
    if not writer:
        return Response("Could not find session", status_code=404)

    payload = await request.json()
    try:
        message = mcp_types.JSONRPCMessage.model_validate(payload)
    except ValidationError as err:
        await writer.send(err)
        return Response("Could not parse message", status_code=400)

    await writer.send(message)
    return Response("Accepted", status_code=202)


@app.get("/mcp/health")
def mcp_health():
    """Health and discovery info for network MCP clients."""
    return {
        "status": "healthy",
        "transport": "sse",
        "sse_path": "/mcp/sse",
        "message_path": "/mcp/messages",
    }

# Mount Gradio UI on the same public ASGI app.
_ui_base_url = os.getenv("DOCMIND_API_URL", "http://127.0.0.1:7860")
app = gr.mount_gradio_app(app, build_demo(base_url=_ui_base_url), path="/")
