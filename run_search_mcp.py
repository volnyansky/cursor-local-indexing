#!/usr/bin/env python3
"""
Call the search_code tool via MCP protocol.
Uses FastMCP Client to connect to the code indexer server and invoke search_code.

Usage:
  # Use default URL (http://localhost:8978/mcp for streamable-http)
  poetry run python run_search_mcp.py

  # Override URL (e.g. if server uses /sse or another host/port)
  MCP_CODE_INDEXER_URL=http://localhost:8978/sse poetry run python run_search_mcp.py

  # Custom query/project via env
  QUERY="pdf" PROJECT="pet" poetry run python run_search_mcp.py
"""
import asyncio
import json
import os
import sys

# streamable-http default path is /mcp; Cursor README uses /sse
MCP_URL = os.getenv("MCP_CODE_INDEXER_URL", "http://localhost:8978/mcp")
QUERY = os.getenv("QUERY", "pdf")
PROJECT = os.getenv("PROJECT", "pet")


async def main():
    from fastmcp import Client
    from fastmcp.client.transports import infer_transport

    transport = infer_transport(MCP_URL)
    client = Client(transport)

    async with client:
        result = await client.call_tool(
            "search_code",
            arguments={"query": QUERY, "project": PROJECT},
            raise_on_error=False,
        )

    # Result is CallToolResult with content (list of ContentBlock)
    if hasattr(result, "content"):
        for block in result.content:
            if hasattr(block, "text"):
                print(block.text)
            elif isinstance(block, dict) and "text" in block:
                print(block["text"])
    elif hasattr(result, "data"):
        print(json.dumps(result.data, indent=2))
    elif hasattr(result, "structured_content") and result.structured_content:
        print(json.dumps(result.structured_content, indent=2))
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
