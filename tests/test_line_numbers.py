"""
Test that process_and_index_documents correctly sets start_line / end_line
for Python files split by CodeSplitter (which does not populate
start_line_number / end_line_number in node metadata).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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

import code_indexer as server  # noqa: E402  (after stubs)


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



# ---------------------------------------------------------------------------
# Fixture: Python source with comments directly above functions, long enough
# that the CodeSplitter chunk boundary falls between the comment and its
# associated function definition.
# ---------------------------------------------------------------------------
COMMENTED_PYTHON_SOURCE = """\
def func_a():
    a1 = 1
    a2 = 2
    a3 = 3
    a4 = 4
    a5 = 5
    a6 = 6
    a7 = 7
    a8 = 8
    a9 = 9
    a10 = 10
    a11 = 11
    a12 = 12
    a13 = 13
    a14 = 14
    a15 = 15
    a16 = 16
    a17 = 17
    a18 = 18
    a19 = 19
    a20 = 20
    a21 = 21
    a22 = 22
    a23 = 23
    a24 = 24
    a25 = 25
    a26 = 26
    a27 = 27
    a28 = 28
    a29 = 29
    a30 = 30
    return a30


# This comment describes func_b.
# It should appear in the same chunk as func_b.
def func_b():
    return 42
"""


class TestCommentReattachment(unittest.TestCase):

    def _run_and_collect_chunks(self, source: str, filename: str):
        """
        Run process_and_index_documents and return (metadatas, texts) tuples
        from every upsert call.
        """
        from llama_index.core import Document

        doc = Document(
            text=source,
            metadata={"file_path": f"/fake/{filename}", "file_name": filename},
        )

        captured_metadatas: list = []
        captured_texts: list = []

        def fake_upsert(ids, documents, metadatas, embeddings=None):
            captured_metadatas.extend(metadatas)
            captured_texts.extend(documents)

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

        return captured_metadatas, captured_texts

    def test_leading_comments_with_function_same_chunk(self):
        """
        Integration test: when CodeSplitter returns nodes where a comment
        trails chunk N and a function definition starts chunk N+1,
        process_and_index_documents must reattach the comment so it appears
        in the same stored chunk as the function — not stranded in the prior chunk.

        The CodeSplitter is mocked to return a deterministic "buggy" split so
        the test does not depend on fragile byte-boundary engineering.
        """
        from llama_index.core import Document
        from llama_index.core.schema import TextNode

        # Simulate nodes as CodeSplitter would produce WITHOUT the fix:
        # node_a ends with a comment, node_b starts with a function definition.
        node_a_text = "def alpha():\n    return 1\n\n\n# Describe beta\n"
        node_a = TextNode(
            text=node_a_text,
            metadata={"file_path": "test.py", "file_name": "test.py"},
        )
        node_a.start_char_idx = 0
        node_a.end_char_idx = len(node_a_text)

        node_b_text = "def beta():\n    return 2\n"
        node_b = TextNode(
            text=node_b_text,
            metadata={"file_path": "test.py", "file_name": "test.py"},
        )
        node_b.start_char_idx = node_a.end_char_idx
        node_b.end_char_idx = node_a.end_char_idx + len(node_b_text)

        full_source = node_a_text + node_b_text
        doc = Document(
            text=full_source,
            metadata={"file_path": "test.py", "file_name": "test.py"},
        )

        captured_texts: list = []

        def fake_upsert(**kwargs):
            captured_texts.extend(kwargs.get("documents", []))

        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = fake_upsert
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        mock_embedding = MagicMock()
        mock_embedding.side_effect = lambda texts: [[0.0] * 4 for _ in texts]

        with (
            patch.object(server, "chroma_client", mock_chroma),
            patch.object(server, "embedding_function", mock_embedding),
            patch("code_indexer.CodeSplitter") as mock_splitter_cls,
        ):
            mock_splitter = MagicMock()
            mock_splitter.get_nodes_from_documents.return_value = [node_a, node_b]
            mock_splitter_cls.return_value = mock_splitter

            server.process_and_index_documents([doc], "test_collection", "chroma_db")

        self.assertEqual(len(captured_texts), 2, "Expected exactly 2 chunks")

        # chunk[0] must no longer end with the comment
        last_nonempty_a = next(
            (l for l in reversed(captured_texts[0].splitlines()) if l.strip()), ""
        )
        self.assertFalse(
            last_nonempty_a.strip().startswith("#"),
            f"chunk[0] still ends with a comment: {captured_texts[0]!r}",
        )

        # chunk[1] must now start with the comment
        first_nonempty_b = next(
            (l for l in captured_texts[1].splitlines() if l.strip()), ""
        )
        self.assertTrue(
            first_nonempty_b.strip().startswith("#"),
            f"chunk[1] does not start with the comment: {captured_texts[1]!r}",
        )

        # chunk[1] must still contain the function definition
        self.assertIn("def beta():", captured_texts[1])

    def test_reattach_leading_comments_unit(self):
        """
        Unit test for reattach_leading_comments: trailing comment on node A
        must be moved to the start of node B when node B begins with a
        function definition.
        """
        from llama_index.core.schema import TextNode

        node_a = TextNode(
            text="def alpha():\n    return 1\n\n\n# Describe beta\n",
            metadata={},
        )
        node_a.start_char_idx = 0
        node_a.end_char_idx = len(node_a.text)

        node_b_text = "def beta():\n    return 2\n"
        node_b = TextNode(text=node_b_text, metadata={})
        node_b.start_char_idx = node_a.end_char_idx
        node_b.end_char_idx = node_a.end_char_idx + len(node_b_text)

        server.reattach_leading_comments([node_a, node_b])

        # The comment must have been moved out of node_a
        last_nonempty_a = next(
            (l for l in reversed(node_a.text.splitlines()) if l.strip()), ""
        )
        self.assertFalse(
            last_nonempty_a.strip().startswith("#"),
            f"node_a still ends with a comment: {node_a.text!r}",
        )

        # The comment must now be at the start of node_b
        first_nonempty_b = next(
            (l for l in node_b.text.splitlines() if l.strip()), ""
        )
        self.assertTrue(
            first_nonempty_b.strip().startswith("#"),
            f"node_b does not start with the comment: {node_b.text!r}",
        )

        # node_b must still contain the function definition
        self.assertIn("def beta():", node_b.text)


if __name__ == "__main__":
    unittest.main()
