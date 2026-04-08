#!/usr/bin/env python3

import asyncio
import logging

from src.code_indexer import initialize_chromadb
from src.observer import index_projects
from src.mcp import mcp

logger = logging.getLogger(__name__)


async def main():
    # Initialize ChromaDB before starting MCP
    success = await initialize_chromadb()

    if success:
        # Start file watching in background (worker task)
        asyncio.create_task(index_projects())
        logger.info("File watching task started")

    await mcp.run_async(transport="http", host="0.0.0.0", port=8978)


if __name__ == "__main__":
    asyncio.run(main())
