# File: mcp_server/server.py
"""MCP Server — exposes DocMind's ingest and query capabilities
as tools that any MCP-compatible AI agent can call."""

import asyncio
import json
import sys
import os

# Add project root to path so app modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.ingest import ingest_file
from app.chain import ask_question

# Create the MCP server instance
server = Server("docmind")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of tools this MCP server exposes."""
    return [
        Tool(
            name="ingest_document",
            description="Use this tool to add a PDF or text document to the knowledge base. Call this before querying.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the PDF or text file to ingest.",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="query_documents",
            description="Use this tool to answer a question using documents in the knowledge base. Pass session_id to maintain conversation continuity across multiple calls in the same agent workflow.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer based on ingested documents.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID. Pass the same session_id across multiple calls to maintain conversation memory. If not provided a new session is created.",
                    },
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route incoming tool calls to the correct RAG function."""
    if name == "ingest_document":
        file_path = arguments.get("file_path", "")
        try:
            chunks = ingest_file(file_path)
            result = {"status": "success", "chunks_added": chunks}
        except Exception as e:
            result = {"status": "error", "message": str(e)}
        return [TextContent(type="text", text=json.dumps(result))]

    elif name == "query_documents":
        question = arguments.get("question", "")
        session_id = arguments.get("session_id", None)
        try:
            answer = ask_question(question=question, session_id=session_id)
            result = {
                "answer": answer["answer"],
                "source_chunks": answer["source_chunks"],
                "relevant_history": answer["relevant_history"],
                "session_id": answer["session_id"],
            }
        except Exception as e:
            result = {"status": "error", "message": str(e)}
        return [TextContent(type="text", text=json.dumps(result))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Start the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
