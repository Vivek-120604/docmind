# File: app/chain.py
"""Chain module — builds a ConversationalRetrievalChain using Groq LLM,
ChromaDB document retriever, and persistent chat history from ChromaDB Cloud."""

import os
import uuid

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from app.retriever import get_retriever
from app.ingest import store_chat_history, retrieve_chat_history

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant. Answer only based on the provided context and conversation history below.
If the answer is not in the context or history, say I don't know.

Previous conversation:
{chat_history}

Context from documents:
{context}

Question:
{question}

Answer:"""


def build_llm():
    """Create a Groq LLM instance using the llama3-8b-8192 model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        model_name="llama3-8b-8192",
        groq_api_key=api_key,
        temperature=0,
    )


def build_chain():
    """Build and return a ConversationalRetrievalChain wired to ChromaDB + Groq."""
    llm = build_llm()
    retriever = get_retriever(k=4)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ask_question(question: str, session_id: str = None) -> dict:
    """Run a question through the conversational RAG chain.

    Retrieves relevant past exchanges from ChromaDB chat_history collection
    filtered by session_id. Stores the new Q&A pair back to ChromaDB after answering.
    Returns answer, source chunks, relevant history, and session_id.
    """
    # Generate a new session_id if one is not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Retrieve relevant past exchanges for this session from ChromaDB Cloud
    past_exchanges = retrieve_chat_history(
        query=question,
        session_id=session_id,
        k=2,
    )

    # Format past exchanges as a list of tuples (human, ai) for LangChain
    chat_history = []
    for exchange in past_exchanges:
        metadata = exchange.get("metadata", {})
        q = metadata.get("question", "")
        a = metadata.get("answer", "")
        if q and a:
            chat_history.append((q, a))

    # Run the chain with document retrieval + conversation history
    chain = build_chain()
    result = chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    answer = result["answer"]

    source_chunks = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in result.get("source_documents", [])
    ]

    # Persist this Q&A pair to ChromaDB chat_history for future queries
    store_chat_history(
        question=question,
        answer=answer,
        session_id=session_id,
    )

    return {
        "answer": answer,
        "source_chunks": source_chunks,
        "relevant_history": past_exchanges,
        "session_id": session_id,
    }
