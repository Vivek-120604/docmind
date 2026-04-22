# File: app/ingest.py
"""Document ingestion — loads PDFs or text files, splits into chunks,
embeds with HuggingFace, and persists to ChromaDB Cloud."""

import os

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


def get_chroma_client():
    """Create a ChromaDB Cloud HttpClient using environment variables."""
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "api.trychroma.com"),
        ssl=True,
        headers={
            "x-chroma-token": os.getenv("CHROMA_API_KEY", ""),
        },
        tenant=os.getenv("CHROMA_TENANT", ""),
        database=os.getenv("CHROMA_DATABASE", ""),
    )


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
    """Full ingestion pipeline: load → split → embed → persist to ChromaDB Cloud.

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
    """Retrieve the most relevant past Q&A pairs for this session from ChromaDB Cloud.
    
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
