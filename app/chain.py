
"""Chain module — builds a LangChain RetrievalQA chain using Groq LLM
and the ChromaDB retriever."""

import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from app.retriever import get_retriever

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant. Answer only based on the provided context. If the answer is not in the context, say I don't know.

Context:
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
    """Build and return a RetrievalQA chain wired to ChromaDB + Groq."""
    llm = build_llm()
    retriever = get_retriever(k=4)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def ask_question(question: str) -> dict:
    """Run a question through the RAG chain.

    Returns a dict with 'answer' and 'source_chunks' keys.
    """
    chain = build_chain()
    result = chain.invoke({"query": question})

    source_chunks = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in result.get("source_documents", [])
    ]

    return {
        "answer": result["result"],
        "source_chunks": source_chunks,
    }
