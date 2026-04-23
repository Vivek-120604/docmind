# File: inspect_db.py
"""Inspector script — connects to ChromaDB Cloud and prints all stored
chunks and chat history so you can verify what is in the database."""

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()


def get_client():
    """Connect to ChromaDB Cloud using credentials from .env file."""
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "api.trychroma.com"),
        ssl=True,
        headers={"x-chroma-token": os.getenv("CHROMA_API_KEY", "")},
        tenant=os.getenv("CHROMA_TENANT", ""),
        database=os.getenv("CHROMA_DATABASE", ""),
    )


def inspect_documents(client):
    """Print all chunks stored in the documents collection."""
    try:
        col = client.get_collection("documents")
        total = col.count()
        print(f"\n{'='*60}")
        print(f"DOCUMENTS COLLECTION — {total} chunks stored")
        print(f"{'='*60}")

        if total == 0:
            print("No chunks found.")
            return

        data = col.get(include=["documents", "metadatas"])
        for i, (doc, meta) in enumerate(
            zip(data["documents"], data["metadatas"])
        ):
            print(f"\n--- Chunk {i+1} of {total} ---")
            print(f"Source : {meta.get('source', 'unknown')}")
            print(f"Page   : {meta.get('page', 'unknown')}")
            print(f"Content: {doc[:400]}")
            print(f"{'─'*40}")

    except Exception as e:
        print(f"Could not read documents collection: {e}")


def inspect_chat_history(client):
    """Print all conversation history stored in the chat_history collection."""
    try:
        col = client.get_collection("chat_history")
        total = col.count()
        print(f"\n{'='*60}")
        print(f"CHAT HISTORY COLLECTION — {total} entries stored")
        print(f"{'='*60}")

        if total == 0:
            print("No history found.")
            return

        data = col.get(include=["documents", "metadatas"])
        for i, (doc, meta) in enumerate(
            zip(data["documents"], data["metadatas"])
        ):
            print(f"\n--- Entry {i+1} of {total} ---")
            print(f"Session  : {meta.get('session_id', 'unknown')}")
            print(f"Timestamp: {meta.get('timestamp', 'unknown')}")
            print(f"Content  : {doc[:400]}")
            print(f"{'─'*40}")

    except Exception as e:
        print(f"Could not read chat_history collection: {e}")


def inspect_embeddings(client):
    """Print embedding dimensions to confirm vectors are being stored."""
    try:
        col = client.get_collection("documents")
        total = col.count()

        if total == 0:
            print("\nNo embeddings to inspect.")
            return

        data = col.get(limit=1, include=["embeddings"])
        if data["embeddings"]:
            dims = len(data["embeddings"][0])
            print(f"\n{'='*60}")
            print(f"EMBEDDINGS — {dims} dimensions per vector")
            print("Model: sentence-transformers/all-MiniLM-L6-v2")
            print("Expected: 384 dimensions")
            print(f"Status: {'✅ Correct' if dims == 384 else '❌ Unexpected'}")
            print(f"{'='*60}")
        else:
            print("\nEmbeddings not returned — may need include=['embeddings']")

    except Exception as e:
        print(f"Could not inspect embeddings: {e}")


if __name__ == "__main__":
    print("Connecting to ChromaDB Cloud...")
    client = get_client()
    print("✅ Connected")

    inspect_documents(client)
    inspect_chat_history(client)
    inspect_embeddings(client)

    print("\n✅ Inspection complete.")
