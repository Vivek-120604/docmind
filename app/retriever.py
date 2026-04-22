"""Retriever module — queries ChromaDB Cloud for the most relevant document chunks."""

from langchain_community.vectorstores import Chroma
from app.ingest import embedding_model, get_chroma_client, COLLECTION_NAME


def get_retriever(k: int = 4):
    """Return a LangChain retriever that fetches the top-k similar chunks."""
    client = get_chroma_client()
    vectorstore = Chroma(
        client=client,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_chunks(query: str, k: int = 4) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query string.

    Returns a list of dicts with 'content' and 'metadata' keys.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
