---
title: DocMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DocMind

**Drop in any document, ask anything — get answers grounded in your files, not hallucinations.**

## Overview

DocMind is a Retrieval-Augmented Generation system that ingests PDF and text documents into a vector database, then answers natural-language questions strictly from the indexed content. It runs as a FastAPI service with a Gradio frontend for humans and MCP (Model Context Protocol) tools for AI agents. Two interfaces, one RAG brain — Claude, custom agents, and IDE copilots can all call `ingest_document` and `query_documents` as first-class tools.

## Architecture

### User Flow
```
┌──────────┐     ┌───────────┐     ┌──────────────────┐     ┌──────────┐
│  User    │────▶│ Gradio UI │────▶│   FastAPI Server  │────▶│ LangChain│
│ (Browser)│     │ (app.py)  │     │   (api.py:7860)   │     │ RAG Chain│
└──────────┘     └───────────┘     └──────────────────┘     └─────┬────┘
                                                                   │
                                                          ┌────────┴────────┐
                                                          │                 │
                                                     ┌────▼─────┐   ┌──────▼──────┐
                                                     │ ChromaDB │   │  Groq LLM   │
                                                     │(Vectors) │   │(llama3-8b)  │
                                                     └──────────┘   └─────────────┘
```

### Agent Flow (MCP)
```
┌───────────┐     ┌────────────┐     ┌───────────┐     ┌──────────┐
│ AI Agent  │────▶│ MCP Server │────▶│ RAG Tools │────▶│ ChromaDB │
│ (Claude,  │     │ (SSE/HTTP) │     │ ingest()  │     │ + Groq   │
│  Custom)  │     │            │     │ query()   │     │  LLM     │
└───────────┘     └────────────┘     └───────────┘     └──────────┘
```

## Tech Stack

| Component        | Technology                             | Purpose                              |
|------------------|----------------------------------------|--------------------------------------|
| LLM              | Groq (llama-3.1-8b-instant)            | Fast inference for answer generation |
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2 | Free, local document embeddings      |
| Vector Store     | ChromaDB                               | Persistent vector storage & retrieval|
| RAG Framework    | LangChain                              | Chain orchestration & retrieval QA   |
| API Server       | FastAPI                                | REST endpoints for ingest & query    |
| MCP Server       | MCP Python SDK (stdio + SSE)           | Tool exposure for AI agents          |
| Frontend         | Gradio                                 | Browser-based upload & Q&A interface |
| Package Manager  | uv                                     | Fast, reproducible dependency management |
| Containerization | Docker                                 | Reproducible deployment              |

## Getting Started

### Using uv (recommended)
```bash
git clone https://github.com/yourusername/DocMind.git
cd DocMind
uv sync
# Edit .env — add your GROQ_API_KEY and ChromaDB credentials
uv run python main.py
```

### Using pip (fallback)
```bash
git clone https://github.com/yourusername/DocMind.git
cd DocMind
pip install -e .
# Edit .env — add your GROQ_API_KEY and ChromaDB credentials
python main.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### Launch the Gradio UI (optional)
```bash
uv run python app.py
```

### Run the MCP Server over stdio (optional)
```bash
uv run python mcp_server/server.py
# Communicates over stdio — connect from any MCP-compatible client
```

### Run network MCP (SSE over HTTP)
```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 7860
# MCP SSE endpoint: GET /mcp/sse
# MCP message endpoint: POST /mcp/messages
```

## Deploy On Hugging Face Spaces

This repository is now configured for a **Docker Space** that runs one unified app on `$PORT` (usually `7860`) serving:
- Gradio UI at `/`
- FastAPI endpoints like `/ingest` and `/query`
- MCP SSE endpoints at `/mcp/sse` and `/mcp/messages`

### 1) Create the Space
- In Hugging Face, create a new **Space**.
- Choose **Docker** as the SDK.
- Push this repository to the Space.

### 2) Configure Space Secrets
In Space **Settings → Variables and secrets**, add:
- `GROQ_API_KEY`
- `CHROMA_API_KEY`
- `CHROMA_TENANT`
- `CHROMA_DATABASE`
- `CHROMA_HOST` (example: `api.trychroma.com`)

### 3) Build and run
No additional startup command is required. The Docker image starts with:
- `space_start.sh` (runs a unified FastAPI app with mounted Gradio + MCP SSE)

After build completes, open your Space URL. The UI should load on the root path.

### 4) Verify deployment
- Open `/` for the Gradio app.
- Open `/mcp/health` to verify network MCP readiness.

## Space Runtime Notes

- `app/api.py` now exposes API routes, mounts Gradio at `/`, and exposes MCP SSE routes.
- `DOCMIND_API_URL` is used by the Gradio callbacks to call the backend.
- `PORT` is used by `space_start.sh` to bind the unified app.

## API Reference

### FastAPI Endpoints

#### `POST /ingest`
Add a document to the knowledge base.

**Request:**
```json
{
  "file_path": "/path/to/document.pdf"
}
```

**Response:**
```json
{
  "status": "success",
  "chunks_added": 42
}
```

#### `POST /query`
Ask a question grounded in ingested documents.

**Request:**
```json
{
  "question": "What is the main conclusion of the report?"
}
```

**Response:**
```json
{
  "answer": "The main conclusion is that...",
  "source_chunks": [
    {
      "content": "...relevant text from the document...",
      "metadata": {"source": "document.pdf", "page": 3}
    }
  ]
}
```

### MCP Tools

#### `ingest_document`
```json
{
  "name": "ingest_document",
  "arguments": {
    "file_path": "/path/to/document.pdf"
  }
}
```
Returns: `{"status": "success", "chunks_added": 42}`

#### `query_documents`
```json
{
  "name": "query_documents",
  "arguments": {
    "question": "What is the main conclusion?"
  }
}
```
Returns: `{"answer": "...", "source_chunks": [...]}`

### MCP Network Endpoints (SSE)

- `GET /mcp/sse`: opens the SSE stream and returns an `endpoint` event with the session-specific POST URL.
- `POST /mcp/messages?session_id=<id>`: sends JSON-RPC messages for that session.
- `GET /mcp/health`: quick transport health check.

## Project Structure

```
DocMind/
├── app/
│   ├── __init__.py          # Package init
│   ├── ingest.py            # Document loading, splitting, embedding
│   ├── retriever.py         # ChromaDB similarity search
│   ├── chain.py             # LangChain RetrievalQA with Groq
│   ├── api.py               # FastAPI + MCP SSE endpoints, Gradio mount
│   └── ui.py                # Shared Gradio UI factory
├── mcp_server/
│   ├── __init__.py          # Package init
│   └── server.py            # MCP server with stdio transport
├── data/
│   └── .gitkeep             # Uploaded documents directory
├── chroma_db/
│   └── .gitkeep             # ChromaDB persistence directory
├── app.py                   # Gradio UI (file upload + Q&A)
├── main.py                  # Uvicorn entry point (port 7860)
├── mcp_config.json          # MCP server metadata & tool definitions
├── .env                     # Environment variables (API keys, ChromaDB config)
├── pyproject.toml           # Project config & dependencies (uv)
├── Dockerfile               # Container build (python:3.11-slim + uv)
└── README.md                # This file
```

## License

MIT
