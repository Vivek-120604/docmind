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
