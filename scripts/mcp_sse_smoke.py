"""Minimal MCP SSE smoke test for DocMind.

Usage:
  python scripts/mcp_sse_smoke.py
  python scripts/mcp_sse_smoke.py --url https://whyvek-docmind.hf.space/mcp/sse
  python scripts/mcp_sse_smoke.py --question "What skills are mentioned?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anyio
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


DEFAULT_URL = "https://whyvek-docmind.hf.space/mcp/sse"
DEFAULT_QUESTION = "What skills are mentioned in the document?"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MCP SSE smoke test against DocMind")
    parser.add_argument("--url", default=DEFAULT_URL, help="MCP SSE endpoint URL")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question for query_documents")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Also call ingest_document with a temporary text file before query",
    )
    return parser


async def _run(url: str, question: str, ingest: bool) -> None:
    temp_file = Path("/tmp/docomind_mcp_smoke.txt")
    temp_file.write_text(
        "Vivek knows Python, FastAPI, LangChain, ChromaDB and Groq.",
        encoding="utf-8",
    )

    print(f"Connecting to: {url}")
    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            if ingest:
                ingest_result = await session.call_tool(
                    "ingest_document", {"file_path": str(temp_file)}
                )
                payload = ingest_result.content[0].text if ingest_result.content else "<empty>"
                print("ingest_document:", payload)

            query_result = await session.call_tool(
                "query_documents", {"question": question}
            )
            payload = query_result.content[0].text if query_result.content else "<empty>"
            print("query_documents:", payload)

            # Print a short parsed summary when payload is JSON text.
            try:
                data = json.loads(payload)
                answer = data.get("answer", "")
                print("Answer:", answer)
            except Exception:
                pass


def main() -> None:
    args = _build_parser().parse_args()
    anyio.run(_run, args.url, args.question, args.ingest)


if __name__ == "__main__":
    main()
