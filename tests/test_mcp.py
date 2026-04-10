"""
Integration tests for the MCP search_code endpoint.

Requirements:
  - Ollama running at localhost:11434 with qwen3-embedding:0.6b pulled
  - No live chroma_db is touched; tests use a temporary directory
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest

import chromadb
from chromadb.config import Settings

_src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
_src_dir_abs = os.path.abspath(_src_dir)
sys.path.insert(0, _src_dir)
import code_indexer as indexer

# Register code_indexer under its package-qualified name so that
# src/mcp.py's "import src.code_indexer as code_indexer" resolves to the
# same module object we configured above.
sys.modules['src.code_indexer'] = indexer
if 'src' not in sys.modules:
    sys.modules['src'] = type(sys)('src')

# Temporarily remove ALL src/ entries from sys.path while loading src/mcp.py
# to prevent 'import mcp' inside fastmcp from resolving to src/mcp.py
# instead of the mcp pip package.
import importlib.util
_saved_paths = [p for p in sys.path if os.path.abspath(p) == _src_dir_abs]
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _src_dir_abs]
try:
    _mcp_path = os.path.join(_src_dir, 'mcp.py')
    _spec = importlib.util.spec_from_file_location("src_mcp", _mcp_path)
    _mcp_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mcp_module)
finally:
    # Restore src/ on sys.path for other test modules that need it.
    for p in _saved_paths:
        sys.path.insert(0, p)
search_code = _mcp_module.search_code

CONTENT_DIR = os.path.join(os.path.dirname(__file__), 'content')
PYTHON_CONTENT = os.path.join(CONTENT_DIR, 'test-content.py')
JS_CONTENT = os.path.join(CONTENT_DIR, 'test-content.js')


def _setup_indexer(tmp_dir: str) -> None:
    """Wire indexer globals to a fresh temp ChromaDB and real Ollama."""
    from chromadb.utils import embedding_functions
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    indexer.embedding_function = embedding_functions.OllamaEmbeddingFunction(
        model_name=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
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


def _run_search(query: str, project: str, n_results: int = 8, threshold: float = 30.0) -> dict:
    """Run search_code synchronously and return parsed JSON result."""
    raw = asyncio.run(search_code(query, project, n_results, threshold))
    return json.loads(raw)


class TestSearchCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.tmp_dir = tempfile.mkdtemp(prefix='chroma_test_mcp_')
            _setup_indexer(cls.tmp_dir)
            _index_file(PYTHON_CONTENT, 'test_search')
            _index_file(JS_CONTENT, 'test_search')
        except Exception as e:
            raise unittest.SkipTest(f"Integration setup failed: {e}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_search_returns_results(self):
        """A basic query must return at least one result."""
        result = _run_search("add two numbers", "search")
        self.assertGreater(result["total_results"], 0)

    def test_result_structure(self):
        """Each result must contain all expected keys."""
        result = _run_search("add two numbers", "search")
        required_keys = {"text", "file_path", "language", "start_line", "end_line", "relevance", "collection"}
        for r in result["results"]:
            self.assertEqual(required_keys, set(r.keys()), f"Unexpected keys: {set(r.keys())}")

    def test_python_language_detected(self):
        """Results from .py files must have language='python'."""
        result = _run_search("calculate sum of two numbers python function", "search")
        py_results = [r for r in result["results"] if r["file_path"].endswith(".py")]
        self.assertTrue(py_results, "No Python results found")
        for r in py_results:
            self.assertEqual(r["language"], "python")

    def test_js_language_detected(self):
        """Results from .js files must have language='javascript'."""
        result = _run_search("class with greet method javascript", "search")
        js_results = [r for r in result["results"] if r["file_path"].endswith(".js")]
        self.assertTrue(js_results, "No JavaScript results found")
        for r in js_results:
            self.assertEqual(r["language"], "javascript")

    def test_start_end_line_valid(self):
        """start_line >= 1 and end_line >= start_line for every result."""
        result = _run_search("function", "search")
        for r in result["results"]:
            self.assertGreaterEqual(r["start_line"], 1, f"start_line < 1: {r}")
            self.assertGreaterEqual(r["end_line"], r["start_line"], f"end_line < start_line: {r}")

    def test_text_matches_line_range(self):
        """For Python results, the returned text must match the file lines at start_line:end_line."""
        result = _run_search("palindrome check", "search")
        py_results = [r for r in result["results"] if r["file_path"] == "test-content.py"]
        self.assertTrue(py_results, "No Python results for palindrome query")

        with open(PYTHON_CONTENT, 'r') as f:
            file_lines = f.readlines()

        for r in py_results:
            region = ''.join(file_lines[r["start_line"] - 1:r["end_line"]])
            for line in r["text"].splitlines():
                if line.strip():
                    self.assertIn(
                        line, region,
                        f"Line {line!r} not found in file lines {r['start_line']}-{r['end_line']}",
                    )

    def test_relevance_ordering(self):
        """Results must be sorted by relevance in descending order."""
        result = _run_search("convert temperature", "search")
        relevances = [r["relevance"] for r in result["results"]]
        self.assertEqual(relevances, sorted(relevances, reverse=True))

    def test_threshold_filtering(self):
        """A very high threshold should return fewer results than a low threshold."""
        low = _run_search("function", "search", threshold=0.0)
        high = _run_search("function", "search", threshold=99.0)
        self.assertGreaterEqual(low["total_results"], high["total_results"])

    def test_add_numbers_in_results(self):
        """Querying for 'python def add_numbers' should return a result containing add_numbers."""
        result = _run_search("python def add_numbers sum", "search")
        self.assertGreater(result["total_results"], 0)
        texts = [r["text"] for r in result["results"]]
        print("Search results texts:", texts)  # Debug print to verify contents
        self.assertTrue(
            any("add_numbers" in t for t in texts),
            "add_numbers not found in any search result",
        )


if __name__ == '__main__':
    unittest.main()
