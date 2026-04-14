#!/usr/bin/env python3

import os
import re
import logging
import json
import traceback
from typing import List, Set
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tree_sitter_language_pack import get_parser

# Import LlamaIndex components
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.schema import TextNode
    print("LlamaIndex dependencies found.")
except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required dependencies:")
    print("pip install llama-index llama-index-readers-file "
          "llama-index-embeddings-huggingface")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from HTTP / client libraries if they are chatty
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Default directories to ignore
DEFAULT_IGNORE_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".ipynb_checkpoints"
}

# Default files to ignore
DEFAULT_IGNORE_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "composer.lock",
    ".DS_Store",
    ".env",
    ".env.local",
    ".env.development",
    ".env.test",
    ".env.production",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    ".coverage",
    "coverage.xml",
    ".eslintcache",
    ".tsbuildinfo"
}

# Default file extensions to include
DEFAULT_FILE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".sql", ".md", ".json", ".yaml", ".yml", ".toml"
    ".tf", ".tpl", ".tfvars"
}

# Single-line comment patterns: # (Python/shell), // (C-family/JS/TS/Go/etc.)
_SINGLE_LINE_COMMENT_RE = re.compile(r'^\s*(#|//)')

# Opening of a block comment (/** or /*) on its own line or starting a line
_BLOCK_COMMENT_OPEN_RE = re.compile(r'^\s*/\*')

# Closing of a block comment (*/)
_BLOCK_COMMENT_CLOSE_RE = re.compile(r'\*/')

# Function / class definition starters (covers Python, JS/TS, Go, Java, C#, etc.)
_DEFINITION_START_RE = re.compile(
    r'^\s*('
    r'def |class |async def |async function |function |func |'
    r'public |private |protected |static |abstract |override |'
    r'export (default )?(function |class |const |let |var |async )|'
    r'const |let |var |'
    r'(pub(\s+fn|\s+async\s+fn))|fn '   # Rust
    r')'
)


def reattach_leading_comments(nodes):
    """
    Post-process CodeSplitter output so that comment lines trailing chunk N
    are moved to the beginning of chunk N+1 when N+1 starts with a
    function/class definition.

    Handles both single-line (# / //) and block (/* ... */ / /** ... */)
    comment styles. Blank lines between the comment and the definition are
    included in the moved block so the pairing stays natural.

    Mutates nodes in-place and returns them.
    """
    for i in range(len(nodes) - 1):
        current = nodes[i]
        nxt = nodes[i + 1]

        lines = current.text.splitlines(keepends=True)
        if not lines:
            continue

        # Find the first non-empty line of the next chunk
        next_first_code = next(
            (l for l in nxt.text.splitlines() if l.strip()), ""
        )
        if not _DEFINITION_START_RE.match(next_first_code):
            continue

        # Walk backwards through current chunk to find the start of the
        # trailing comment block (single-line or block comment).
        split_idx = len(lines)
        j = len(lines) - 1

        # Skip trailing blank lines first
        while j >= 0 and not lines[j].strip():
            j -= 1

        if j < 0:
            continue

        # Collect backwards while we're inside comment lines
        in_block_comment = False
        while j >= 0:
            line = lines[j]
            stripped = line.strip()

            if not stripped:
                # blank line — stop; don't pull blank separator lines
                break

            if _BLOCK_COMMENT_CLOSE_RE.search(line):
                # End of a block comment (reading backwards = we enter it)
                in_block_comment = True
                split_idx = j
                j -= 1
                continue

            if in_block_comment:
                split_idx = j
                if _BLOCK_COMMENT_OPEN_RE.match(line):
                    in_block_comment = False
                j -= 1
                continue

            if _SINGLE_LINE_COMMENT_RE.match(line):
                split_idx = j
                j -= 1
                continue

            # Non-comment, non-blank line — stop
            break

        if split_idx == len(lines):
            continue  # nothing to move

        # Guard: don't leave current chunk empty
        has_non_comment = any(
            l.strip() and not _SINGLE_LINE_COMMENT_RE.match(l)
            and not _BLOCK_COMMENT_OPEN_RE.match(l)
            for l in lines[:split_idx]
        )
        if not has_non_comment:
            continue

        comment_block = "".join(lines[split_idx:])
        current.text = "".join(lines[:split_idx])
        nxt.text = comment_block + nxt.text

        moved = len(comment_block)
        if current.end_char_idx is not None:
            current.end_char_idx -= moved
        if nxt.start_char_idx is not None:
            nxt.start_char_idx -= moved

    return nodes


# Global variables
config = None
chroma_client = None
embedding_function = None


def get_config_from_env():
    """Get configuration from environment variables."""
    projects_root = os.getenv("PROJECTS_ROOT", "/projects")
    folders_to_index = os.getenv("FOLDERS_TO_INDEX", "").split(",")
    folders_to_index = [f.strip() for f in folders_to_index if f.strip()]

    # Get additional ignore dirs and files from environment
    additional_ignore_dirs = os.getenv(
        "ADDITIONAL_IGNORE_DIRS", ""
    ).split(",")
    additional_ignore_dirs = [
        d.strip() for d in additional_ignore_dirs if d.strip()
    ]

    additional_ignore_files = os.getenv(
        "ADDITIONAL_IGNORE_FILES", ""
    ).split(",")
    additional_ignore_files = [
        f.strip() for f in additional_ignore_files if f.strip()
    ]

    # Combine default and additional ignore patterns
    ignore_dirs = list(DEFAULT_IGNORE_DIRS | set(additional_ignore_dirs))
    ignore_files = list(DEFAULT_IGNORE_FILES | set(additional_ignore_files))

    if not folders_to_index:
        logger.warning("No folders specified to index. Using root directory.")
        folders_to_index = [""]

    return {
        "projects_root": projects_root,
        "folders_to_index": folders_to_index,
        "ignore_dirs": ignore_dirs,
        "ignore_files": ignore_files,
        "file_extensions": list(DEFAULT_FILE_EXTENSIONS)
    }


async def initialize_chromadb():
    """Initialize ChromaDB and embedding function asynchronously."""
    global config, chroma_client, embedding_function

    try:
        # Get configuration from environment
        config = get_config_from_env()
        logger.info("Configuration loaded successfully")

        # Initialize ChromaDB client with telemetry disabled
        chroma_client = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info("ChromaDB client initialized")

        # Initialize embedding function using Ollama with the qwen3-embedding model
        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )
        embedding_function = embedding_functions.OllamaEmbeddingFunction(
            model_name= os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            url=ollama_base_url,
        )
        logger.info(
            f"Embedding function initialized with Ollama model {os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")} "
            f"at {ollama_base_url}"
        )

        return True
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        # Still need to assign default values to global variables
        if config is None:
            config = {"projects_root": "", "folders_to_index": [""]}
        if chroma_client is None:
            # Create an empty client as a fallback
            try:
                chroma_client = chromadb.PersistentClient(path="chroma_db")
            except Exception as db_err:
                logger.error(
                    f"Failed to create fallback ChromaDB client: {db_err}"
                )
        if embedding_function is None:
            try:
                ollama_base_url = os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434"
                )
                embedding_function = embedding_functions.OllamaEmbeddingFunction(
                    model_name="qwen3-embedding:0.6b",
                    base_url=ollama_base_url,
                )
            except Exception as embed_err:
                logger.error(
                    f"Failed to create fallback embedding function: "
                    f"{embed_err}"
                )
        return False


def sanitize_collection_name(folder_name: str) -> str:
    """Convert folder name to a valid collection name by replacing forward slashes with underscores."""
    return folder_name.replace("/", "_")


def is_valid_file(
    file_path: str,
    ignore_dirs: Set[str],
    file_extensions: Set[str],
    ignore_files: Set[str] = None
) -> bool:
    """Check if a file should be processed based on its path and extension."""
    # Check if path contains ignored directory
    parts = file_path.split(os.path.sep)
    for part in parts:
        if part in ignore_dirs:
            return False

    # Get file name and check against ignored files
    file_name = os.path.basename(file_path)

    # Use provided ignore_files or fall back to default
    files_to_ignore = ignore_files if ignore_files is not None else DEFAULT_IGNORE_FILES

    # Check exact matches
    if file_name in files_to_ignore:
        return False

    # Check wildcard patterns
    for pattern in files_to_ignore:
        if pattern.startswith("*"):
            if file_name.endswith(pattern[1:]):
                return False

    # Check file extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in file_extensions if file_extensions else True


def load_documents(
    directory: str,
    ignore_dirs: Set[str] = DEFAULT_IGNORE_DIRS,
    file_extensions: Set[str] = DEFAULT_FILE_EXTENSIONS,
    ignore_files: Set[str] = None
) -> List[Document]:
    """Load documents from a directory, filtering out ignored paths."""
    try:
        # Get all files recursively
        all_files = []
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in ignore_dirs and not d.startswith('.')
            ]

            for file in files:
                abs_file_path = os.path.join(root, file)
                if is_valid_file(
                    abs_file_path,
                    ignore_dirs,
                    file_extensions,
                    ignore_files
                ):
                    # Calculate relative path from the directory being indexed
                    rel_file_path = os.path.relpath(abs_file_path, directory)
                    all_files.append((abs_file_path, rel_file_path))

        if not all_files:
            logger.warning(f"No valid files found in {directory}")
            return []

        # Load the filtered files using absolute paths for reading
        reader = SimpleDirectoryReader(
            input_files=[abs_path for abs_path, _ in all_files],
            exclude_hidden=True
        )
        documents = reader.load_data()

        # Update the metadata to use relative paths
        for doc, (_, rel_path) in zip(documents, all_files):
            doc.metadata["file_path"] = rel_path

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def process_and_index_documents(
    documents: List[Document],
    collection_name: str,
    persist_directory: str
) -> None:
    """Process documents with CodeSplitter and index them in ChromaDB."""
    if not documents:
        logger.warning("No documents to process.")
        return

    try:
        # Try to get collection if it exists or create a new one
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return

    # Process each document
    total_nodes = 0
    total_documents = len(documents)

    for doc_index, doc in enumerate(documents, start=1):
        try:
            # Extract file path from metadata
            file_path = doc.metadata.get("file_path", "unknown")
            file_name = os.path.basename(file_path)

            # Determine language from file extension
            _, ext = os.path.splitext(file_name)
            language = ext[1:] if ext else "text"  # Remove the dot

            # Handle Markdown and other text files differently
            code_file_extensions = [
                "py", "python", "js", "jsx", "ts", "tsx", "java", "c",
                "cpp", "h", "hpp", "cs", "go", "rb", "php", "swift",
                "kt", "rs", "scala"
            ]

            if language in code_file_extensions:
                # Determine parser language based on file extension
                parser_language = "python"  # Default fallback
                if language in ["py", "python"]:
                    parser_language = "python"
                elif language in ["js", "jsx"]:
                    parser_language = "javascript"
                elif language in [ "ts", "tsx"]:
                    parser_language = "typescript"
                elif language in ["java"]:
                    parser_language = "java"
                elif language in ["c", "cpp", "h", "hpp"]:
                    parser_language = "cpp"
                elif language in ["cs"]:
                    parser_language = "csharp"
                elif language in ["go"]:
                    parser_language = "go"
                elif language in ["rb"]:
                    parser_language = "ruby"
                elif language in ["php"]:
                    parser_language = "php"
                elif language in ["swift"]:
                    parser_language = "swift"
                elif language in ["kt"]:
                    parser_language = "kotlin"
                elif language in ["rs"]:
                    parser_language = "rust"
                elif language in ["scala"]:
                    parser_language = "scala"

                # Create parser and splitter for this specific language
                try:
                    code_parser = get_parser(parser_language)
                    splitter = CodeSplitter(
                        language=parser_language,
                        chunk_lines=40,
                        chunk_lines_overlap=15,
                        max_chars=1500,
                        parser=code_parser
                    )
                    nodes = splitter.get_nodes_from_documents([doc])
                    reattach_leading_comments(nodes)
                except Exception as e:
                    logger.warning(
                        f"Could not create parser for {parser_language}, "
                        f"falling back to text-based splitting: {e}"
                    )
                    # Fall back to text-based splitting
                    nodes = []
                    lines = doc.text.split("\n")
                    chunk_size = 40
                    overlap = 15

                    for i in range(0, len(lines), chunk_size - overlap):
                        start_idx = i
                        end_idx = min(i + chunk_size, len(lines))

                        if start_idx >= len(lines):
                            continue

                        chunk_text = "\n".join(lines[start_idx:end_idx])

                        if not chunk_text.strip():
                            continue

                        node = TextNode(
                            text=chunk_text,
                            metadata={
                                "start_line_number": start_idx + 1,
                                "end_line_number": end_idx,
                                "file_path": file_path,
                                "file_name": file_name,
                            }
                        )
                        nodes.append(node)
            else:
                # For non-code files, manually split by lines
                nodes = []
                lines = doc.text.split("\n")
                chunk_size = 40
                overlap = 15

                for i in range(0, len(lines), chunk_size - overlap):
                    start_idx = i
                    end_idx = min(i + chunk_size, len(lines))

                    if start_idx >= len(lines):
                        continue

                    chunk_text = "\n".join(lines[start_idx:end_idx])

                    if not chunk_text.strip():
                        continue

                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            "start_line_number": start_idx + 1,
                            "end_line_number": end_idx,
                            "file_path": file_path,
                            "file_name": file_name,
                        }
                    )
                    nodes.append(node)

            # Filter out degenerate tiny chunks (e.g. lone brackets from tree-sitter)
            MIN_CHUNK_CHARS = 20
            def is_meaningful_chunk(node):
                stripped = node.text.strip()
                if len(stripped) < MIN_CHUNK_CHARS:
                    return False
                # Filter chunks that are mostly braces/whitespace
                code_chars = stripped.translate(str.maketrans("", "", "{}()[];\n\r\t "))
                return len(code_chars) >= MIN_CHUNK_CHARS
            nodes = [n for n in nodes if is_meaningful_chunk(n)]

            if not nodes:
                logger.warning(f"No nodes generated for {file_path}")
                continue

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, node in enumerate(nodes):
                start_line = node.metadata.get("start_line_number", 0)
                end_line = node.metadata.get("end_line_number", 0)

                if start_line == 0 or end_line == 0:
                    if node.start_char_idx is not None and node.end_char_idx is not None:
                        start_line = doc.text[:node.start_char_idx].count("\n") + 1
                        end_line = doc.text[:node.end_char_idx].count("\n") + 1
                    else:
                        start_line = 1
                        end_line = len(node.text.split("\n"))

                chunk_id = f"{file_path}_{start_line}_{end_line}_{i}"

                metadata = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "language": parser_language,
                    "start_line": start_line,
                    "end_line": end_line,
                }

                ids.append(chunk_id)
                texts.append(node.text)
                metadatas.append(metadata)

            total_nodes += len(nodes)
            # Add nodes to ChromaDB collection
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            # Progress output after upsert completes so it stays visible
            print(f"\r{doc_index} / {total_documents} : {file_path}", end="", flush=True)


           

        except Exception as e:
            logger.error(
                f"Error processing document "
                f"{doc.metadata.get('file_path', 'unknown')}: {e}"
            )

    # Ensure we end the progress line cleanly
    if documents:
        print()

    logger.info(
        f"Successfully indexed {total_nodes} code chunks "
        f"across {len(documents)} files"
    )


async def perform_initial_indexing(folder: str) -> bool:
    """Check if collection exists and perform initial indexing if needed."""
    try:
        folder_path = os.path.join(config["projects_root"], folder)
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return False

        collection_name = sanitize_collection_name(folder)

        # Check if collection exists
        try:
            chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Collection {collection_name} already exists, skipping initial indexing")
            return True
        except Exception:
            logger.info(f"Collection {collection_name} not found, performing initial indexing")

        # Load and process all documents in the folder
        documents = load_documents(
            folder_path,
            ignore_dirs=set(config["ignore_dirs"]),
            file_extensions=set(config["file_extensions"]),
            ignore_files=set(config["ignore_files"])
        )

        if documents:
            process_and_index_documents(documents, collection_name, "chroma_db")
            logger.info(f"Successfully performed initial indexing for {folder}")
            return True
        else:
            logger.warning(f"No documents found to index in {folder}")
            return False

    except Exception as e:
        logger.error(f"Error during initial indexing of {folder}: {e}")
        return False
