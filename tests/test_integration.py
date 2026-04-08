"""
Integration tests for code_indexer.py.

Requirements:
  - Ollama running at localhost:11434 with qwen3-embedding:0.6b pulled
  - No live chroma_db is touched; tests use a temporary directory
"""

import asyncio
import os
import shutil
import sys
import tempfile
import unittest

import chromadb
from chromadb.config import Settings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import code_indexer as indexer

CONTENT_DIR = os.path.join(os.path.dirname(__file__), 'content')
PYTHON_CONTENT = os.path.join(CONTENT_DIR, 'test-content.py')
JS_CONTENT = os.path.join(CONTENT_DIR, 'test-content.js')
LONG_PYTHON_CONTENT = os.path.join(CONTENT_DIR, 'test-content-long.py')


def _setup_indexer(tmp_dir: str) -> None:
    """Wire indexer globals to a fresh temp ChromaDB and real Ollama."""
    asyncio.run(indexer.initialize_chromadb())
    # Override client to use isolated temp directory
    indexer.chroma_client = chromadb.PersistentClient(
        path=tmp_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _index_file(file_path: str, collection_name: str) -> None:
    """Load a single file and index it into the named collection."""
    from llama_index.core import Document

    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    file_name = os.path.basename(file_path)
    rel_path = os.path.relpath(file_path, CONTENT_DIR)
    doc = Document(
        text=source,
        metadata={"file_path": rel_path, "file_name": file_name},
    )
    indexer.process_and_index_documents([doc], collection_name, "")


def _get_all_chunks(collection_name: str):
    """Return (documents, metadatas) lists for every chunk in the collection."""
    col = indexer.chroma_client.get_collection(
        name=collection_name,
        embedding_function=indexer.embedding_function,
    )
    result = col.get(include=['documents', 'metadatas'])
    return result['documents'], result['metadatas']


if __name__ == '__main__':
    unittest.main()
