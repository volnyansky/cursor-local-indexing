"""
Test that process_and_index_documents correctly sets start_line / end_line
for Python files split by CodeSplitter (which does not populate
start_line_number / end_line_number in node metadata).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so we can import code_indexer_server without a running
# ChromaDB instance or a loaded embedding model.
# ---------------------------------------------------------------------------

# Stub out chromadb at the module level before the import
chroma_stub = MagicMock()
sys.modules.setdefault("chromadb", chroma_stub)
sys.modules.setdefault("chromadb.config", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

# fastmcp stub
fastmcp_stub = MagicMock()
fastmcp_stub.FastMCP = MagicMock(return_value=MagicMock())
sys.modules.setdefault("fastmcp", fastmcp_stub)

# watchdog stubs
sys.modules.setdefault("watchdog", MagicMock())
sys.modules.setdefault("watchdog.observers", MagicMock())
sys.modules.setdefault("watchdog.events", MagicMock())

import code_indexer_server as server  # noqa: E402  (after stubs)


# ---------------------------------------------------------------------------
# A multi-function Python snippet long enough to produce several chunks.
# Functions are placed so that they start on known line numbers.
# ---------------------------------------------------------------------------
# Source is intentionally long (>80 lines) to force CodeSplitter to produce
# multiple chunks (chunk_lines=40 in process_and_index_documents).
PYTHON_SOURCE = """\
def alpha():
    x = 1
    y = 2
    z = 3
    w = x + y + z
    return w


def beta():
    a = "hello"
    b = "world"
    c = a + " " + b
    return c


def gamma():
    items = list(range(10))
    total = sum(items)
    average = total / len(items)
    return average


def delta():
    data = {"key": "value"}
    data["extra"] = 42
    data["more"] = [1, 2, 3]
    return data


def epsilon():
    result = []
    for i in range(10):
        if i % 2 == 0:
            result.append(i * 2)
        else:
            result.append(i)
    return result


def zeta():
    mapping = {}
    for k in range(5):
        mapping[k] = k ** 2
    return mapping


def eta():
    values = [10, 20, 30, 40, 50]
    minimum = min(values)
    maximum = max(values)
    return minimum, maximum


def theta():
    x = 100
    y = 200
    z = x * y
    return z


def iota():
    text = "iota"
    upper = text.upper()
    lower = text.lower()
    return upper, lower


def kappa():
    total = 0
    for i in range(20):
        total += i
    return total


def lambda_func():
    pairs = [(1, "a"), (2, "b"), (3, "c")]
    result = {k: v for k, v in pairs}
    return result


def mu():
    stack = []
    stack.append(1)
    stack.append(2)
    stack.append(3)
    top = stack.pop()
    return top


def nu():
    numbers = list(range(1, 11))
    squares = [n ** 2 for n in numbers]
    return squares


def xi():
    s = set()
    for i in range(5):
        s.add(i)
        s.add(i * 2)
    return s


def omicron():
    a = 1
    b = 2
    c = 3
    return a + b + c


def pi_func():
    import math
    return math.pi


def rho():
    words = ["foo", "bar", "baz"]
    joined = ", ".join(words)
    return joined
"""


class TestLineNumbers(unittest.TestCase):

    def _run_and_collect_metadatas(self, source: str, filename: str) -> list:
        """
        Run process_and_index_documents with a mocked ChromaDB collection and
        return all metadata dicts that were passed to collection.upsert().
        """
        from llama_index.core import Document

        doc = Document(
            text=source,
            metadata={"file_path": f"/fake/{filename}", "file_name": filename},
        )

        # Capture every upsert call
        captured_metadatas: list = []

        def fake_upsert(ids, documents, metadatas, embeddings=None):
            captured_metadatas.extend(metadatas)

        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = fake_upsert

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection

        mock_embedding = MagicMock()
        mock_embedding.side_effect = lambda texts: [[0.0] * 4 for _ in texts]

        with (
            patch.object(server, "chroma_client", mock_chroma),
            patch.object(server, "embedding_function", mock_embedding),
        ):
            server.process_and_index_documents([doc], "test_collection", "chroma_db")

        return captured_metadatas

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_python_chunks_have_nonzero_start_lines(self):
        """Every chunk must have start_line >= 1 (never 0)."""
        metadatas = self._run_and_collect_metadatas(PYTHON_SOURCE, "sample.py")
        self.assertTrue(len(metadatas) > 0, "Expected at least one chunk")
        for m in metadatas:
            self.assertGreaterEqual(m["start_line"], 1, f"start_line < 1: {m}")

    def test_python_chunks_start_line_not_all_one(self):
        """
        The original bug: every chunk had start_line=1.
        With the fix, later chunks must start after line 1.
        """
        metadatas = self._run_and_collect_metadatas(PYTHON_SOURCE, "sample.py")
        self.assertTrue(len(metadatas) > 1, "Need >1 chunk to test line progression")
        start_lines = [m["start_line"] for m in metadatas]
        self.assertTrue(
            any(sl > 1 for sl in start_lines),
            f"All chunks have start_line=1 — bug is still present. start_lines={start_lines}",
        )

    def test_python_chunks_end_line_greater_than_start_line(self):
        """end_line must be >= start_line for every chunk."""
        metadatas = self._run_and_collect_metadatas(PYTHON_SOURCE, "sample.py")
        for m in metadatas:
            self.assertGreaterEqual(
                m["end_line"],
                m["start_line"],
                f"end_line < start_line: {m}",
            )

    def test_python_chunks_lines_increase_monotonically(self):
        """
        Chunks should be ordered — each chunk's start_line must be >= the
        previous chunk's start_line.
        """
        metadatas = self._run_and_collect_metadatas(PYTHON_SOURCE, "sample.py")
        start_lines = [m["start_line"] for m in metadatas]
        for i in range(1, len(start_lines)):
            self.assertGreaterEqual(
                start_lines[i],
                start_lines[i - 1],
                f"Chunks are not ordered by line number: {start_lines}",
            )

    def test_python_last_chunk_end_line_near_eof(self):
        """The last chunk's end_line should be close to the total line count."""
        metadatas = self._run_and_collect_metadatas(PYTHON_SOURCE, "sample.py")
        total_lines = len(PYTHON_SOURCE.splitlines())
        last_end = metadatas[-1]["end_line"]
        self.assertGreaterEqual(
            last_end,
            total_lines // 2,
            f"Last chunk end_line={last_end} seems too small for {total_lines}-line file",
        )


if __name__ == "__main__":
    unittest.main()
