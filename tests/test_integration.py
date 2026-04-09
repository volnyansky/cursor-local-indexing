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
    from chromadb.utils import embedding_functions
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    indexer.embedding_function = embedding_functions.OllamaEmbeddingFunction(
        model_name = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        url=ollama_base_url,
    )
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
    """Return (documents, metadatas) lists for every chunk in the collection, sorted by ID."""
    col = indexer.chroma_client.get_collection(
        name=collection_name,
        embedding_function=indexer.embedding_function,
    )
    result = col.get(include=['documents', 'metadatas'])
    combined = sorted(zip(result['ids'], result['documents'], result['metadatas']))
    docs = [d for _, d, _ in combined]
    metas = [m for _, _, m in combined]
    return docs, metas


class TestPythonFileIndexing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.tmp_dir = tempfile.mkdtemp(prefix='chroma_test_py_')
            _setup_indexer(cls.tmp_dir)
            _index_file(PYTHON_CONTENT, 'test_py')
            cls.docs, cls.metas = _get_all_chunks('test_py')
        except Exception as e:
            raise unittest.SkipTest(f"Integration setup failed: {e}")

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
            self.assertTrue(
                any(comment_prefix in d for d in matching),
                f"Comment {comment_prefix!r} not in any chunk containing {func_sig!r}",
            )

    def test_line_numbers_accurate(self):
        with open(PYTHON_CONTENT, 'r') as f:
            file_lines = f.readlines()

        for doc, meta in zip(self.docs, self.metas):
            start = meta['start_line']
            end = meta['end_line']
            self.assertGreaterEqual(start, 1, f"start_line < 1: {meta}")
            self.assertGreaterEqual(end, start, f"end_line < start_line: {meta}")
            self.assertLessEqual(end, len(file_lines), f"end_line beyond EOF: {meta}")
            region = ''.join(file_lines[start - 1:end])
            # Every non-empty line in the chunk must appear in the file region
            for line in doc.splitlines():
                if line.strip():
                    self.assertIn(
                        line, region,
                        f"Line {line!r} from chunk not found in file lines {start}-{end}",
                    )


class TestJSFileIndexing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.tmp_dir = tempfile.mkdtemp(prefix='chroma_test_js_')
            _setup_indexer(cls.tmp_dir)
            _index_file(JS_CONTENT, 'test_js')
            cls.docs, cls.metas = _get_all_chunks('test_js')
        except Exception as e:
            raise unittest.SkipTest(f"Integration setup failed: {e}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _chunks_containing(self, text: str):
        return [d for d in self.docs if text in d]

    def test_all_functions_indexed(self):
        for symbol in ('testFunc', 'testMyClass', 'MyClass'):
            self.assertTrue(
                self._chunks_containing(symbol),
                f"No chunk found containing '{symbol}'",
            )

    def test_jsdoc_with_named_function(self):
        """function testFunc() must be in the same chunk as its JSDoc."""
        matching = self._chunks_containing('function testFunc()')
        self.assertTrue(matching, "No chunk for 'function testFunc()'")
        self.assertTrue(
            any('/**' in d for d in matching),
            f"JSDoc '/**' not found in any chunk with testFunc:\n{matching}",
        )

    def test_jsdoc_with_class(self):
        """class MyClass must be in the same chunk as its JSDoc."""
        matching = self._chunks_containing('class MyClass')
        self.assertTrue(matching, "No chunk for 'class MyClass'")
        self.assertTrue(
            any('/**' in d for d in matching),
            f"JSDoc '/**' not found in any chunk with MyClass:\n{matching}",
        )


class TestCommentReattachmentIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.tmp_dir = tempfile.mkdtemp(prefix='chroma_test_long_')
            _setup_indexer(cls.tmp_dir)
            _index_file(LONG_PYTHON_CONTENT, 'test_long')
            cls.docs, cls.metas = _get_all_chunks('test_long')
        except Exception as e:
            raise unittest.SkipTest(f"Integration setup failed: {e}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_func_b_comment_is_in_func_b_chunk(self):
        """The comment 'describes func_b' must be in the same chunk as def func_b."""
        func_b_chunks = [d for d in self.docs if 'def func_b' in d]
        self.assertTrue(func_b_chunks, "No chunk contains 'def func_b'")
        self.assertTrue(
            any('# This comment describes func_b.' in d for d in func_b_chunks),
            f"Comment not found in any func_b chunk:\n{func_b_chunks}",
        )

    def test_func_b_comment_not_in_func_a_only_chunk(self):
        """The func_a chunk (without func_b) must NOT contain the func_b comment."""
        func_a_only_chunks = [
            d for d in self.docs
            if 'def func_a' in d and 'def func_b' not in d
        ]
        self.assertTrue(func_a_only_chunks, "No chunk contains 'def func_a' alone")
        for chunk in func_a_only_chunks:
            self.assertNotIn(
                '# This comment describes func_b.',
                chunk,
                f"func_b comment leaked into func_a chunk:\n{chunk}",
            )


if __name__ == '__main__':
    unittest.main()
