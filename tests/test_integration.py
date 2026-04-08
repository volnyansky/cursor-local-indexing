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


class TestPythonFileIndexing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix='chroma_test_py_')
        _setup_indexer(cls.tmp_dir)
        _index_file(PYTHON_CONTENT, 'test_py')
        cls.docs, cls.metas = _get_all_chunks('test_py')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _chunks_containing(self, text: str):
        return [d for d in self.docs if text in d]

    def test_all_functions_indexed(self):
        for fn in ('add_numbers', 'is_palindrome', 'find_max', 'celsius_to_fahrenheit'):
            self.assertTrue(
                self._chunks_containing(f'def {fn}'),
                f"No chunk found containing 'def {fn}'",
            )

    def test_each_function_chunk_contains_its_comment(self):
        cases = [
            ('def add_numbers', '# Function 1:'),
            ('def is_palindrome', '# Function 2:'),
            ('def find_max', '# Function 3:'),
            ('def celsius_to_fahrenheit', '# Function 4:'),
        ]
        for func_sig, comment_prefix in cases:
            matching = [d for d in self.docs if func_sig in d]
            self.assertTrue(matching, f"No chunk for {func_sig!r}")
            chunk = matching[0]
            self.assertIn(
                comment_prefix, chunk,
                f"Comment {comment_prefix!r} not in chunk containing {func_sig!r}:\n{chunk}",
            )

    def test_line_numbers_accurate(self):
        with open(PYTHON_CONTENT, 'r') as f:
            file_lines = f.readlines()

        for doc, meta in zip(self.docs, self.metas):
            start = meta['start_line']
            end = meta['end_line']
            self.assertGreaterEqual(start, 1, f"start_line < 1: {meta}")
            self.assertGreaterEqual(end, start, f"end_line < start_line: {meta}")
            self.assertLessEqual(end, len(file_lines) + 1, f"end_line beyond EOF: {meta}")
            region = ''.join(file_lines[start - 1:end])
            # Every non-empty line in the chunk must appear in the file region
            for line in doc.splitlines():
                if line.strip():
                    self.assertIn(
                        line.strip(), region,
                        f"Line {line!r} from chunk not found in file lines {start}-{end}",
                    )


if __name__ == '__main__':
    unittest.main()
