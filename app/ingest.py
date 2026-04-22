# File: app/ingest.py
"""Document ingestion — loads files, chunks text, embeds content,
and persists vectors to ChromaDB (Cloud with local fallback)."""

import os
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

COLLECTION_NAME = "documents"
CHAT_HISTORY_COLLECTION = "chat_history"

# Shared embedding model — loaded once, reused everywhere
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def _truthy(value: str | None) -> bool:
    """Parse common truthy environment variable values."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_chroma_client():
    """Create a Chroma client, preferring Cloud with automatic local fallback.

    Cloud mode is used when CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE
    are available and valid. If Cloud auth fails, we transparently fall back to a
    local persistent ChromaDB to keep ingestion/query functional in containerized
    runtimes (for example when secrets are missing or misconfigured).
    """
    force_local = _truthy(os.getenv("CHROMA_USE_LOCAL"))
    api_key = os.getenv("CHROMA_API_KEY", "")
    tenant = os.getenv("CHROMA_TENANT", "")
    database = os.getenv("CHROMA_DATABASE", "")
    cloud_host = os.getenv("CHROMA_HOST", "api.trychroma.com")

    has_cloud_config = all([api_key, tenant, database]) and not force_local

    if has_cloud_config:
        try:
            client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=api_key,
                cloud_host=cloud_host,
            )
            # Validate credentials once at startup/use so permission errors
            # can gracefully switch to local storage.
            client.list_collections()
            return client
        except Exception as e:
            print(f"[DocMind] Chroma Cloud unavailable, using local fallback: {e}")

    local_path = os.getenv("CHROMA_LOCAL_PATH", "/tmp/chroma")
    Path(local_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=local_path)


def load_document(file_path: str):
    """Load a PDF or text file and return a list of LangChain Documents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .txt, or .md")

    return loader.load()


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def ingest_file(file_path: str) -> int:
    """Full ingestion pipeline: load → split → embed → persist to ChromaDB.

    Returns the number of chunks added to the vector store.
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)

    client = get_chroma_client()

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    return len(chunks)


def store_chat_history(question: str, answer: str, session_id: str) -> None:
    """Store a Q&A pair as an embedding in the chat_history ChromaDB collection.
    
    This gives the RAG persistent memory of past conversations across restarts.
    Each entry is stored with its session_id so agents can retrieve session-specific history.
    """
    from langchain_community.vectorstores import Chroma
    import datetime

    client = get_chroma_client()

    # Format the Q&A pair as a single document for embedding
    document_text = f"Q: {question}\nA: {answer}"

    # Create a LangChain Document with session metadata
    from langchain.schema import Document
    doc = Document(
        page_content=document_text,
        metadata={
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    )

    Chroma.from_documents(
        documents=[doc],
        embedding=embedding_model,
        client=client,
        collection_name=CHAT_HISTORY_COLLECTION,
    )


def retrieve_chat_history(query: str, session_id: str, k: int = 2) -> list[dict]:
    """Retrieve the most relevant past Q&A pairs for this session from ChromaDB.
    
    Filters by session_id so each agent workflow only sees its own history.
    Returns top-k most semantically relevant past exchanges.
    """
    from langchain_community.vectorstores import Chroma

    client = get_chroma_client()

    vectorstore = Chroma(
        client=client,
        embedding_function=embedding_model,
        collection_name=CHAT_HISTORY_COLLECTION,
    )

    # Filter by session_id so sessions don't bleed into each other
    results = vectorstore.similarity_search(
        query,
        k=k,
        filter={"session_id": session_id},
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in results
    ]
