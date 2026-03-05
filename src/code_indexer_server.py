#!/usr/bin/env python3

import os
import logging
import json
import traceback
from typing import List, Set
from pydantic.v1.networks import host_regex
from typing_extensions import Annotated
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tree_sitter_language_pack import get_parser
import asyncio
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Import LlamaIndex components
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core import SimpleDirectoryReader
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
    ".tf",".tpl",".tfvars"
}

# Global variables
config = None
chroma_client = None
embedding_function = None
mcp = FastMCP(name="Code Indexer Server")
observers = []


@mcp.custom_route("/rebuild/{project_name}", methods=["POST"])
async def rebuild_index(request: Request) -> JSONResponse:
    project_name = request.path_params["project_name"]

    # Find folder matching project_name (same logic as search_code)
    matching_folder = None
    for folder in config["folders_to_index"]:
        collection_name = sanitize_collection_name(folder)
        if collection_name.lower().split("_")[-1] == project_name.lower():
            matching_folder = folder
            break

    if not matching_folder:
        return JSONResponse(
            {"error": f"Project '{project_name}' not found in configured folders"},
            status_code=404
        )

    collection_name = sanitize_collection_name(matching_folder)
    folder_path = os.path.join(config["projects_root"], matching_folder)

    def do_rebuild():
        try:
            chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name} for rebuild")
        except Exception:
            pass  # Collection may not exist yet

        documents = load_documents(
            folder_path,
            ignore_dirs=set(config["ignore_dirs"]),
            file_extensions=set(config["file_extensions"]),
            ignore_files=set(config["ignore_files"])
        )

        if not documents:
            logger.warning(f"No indexable documents found in {folder_path}")
            return

        process_and_index_documents(documents, collection_name, "chroma_db")
        logger.info(f"Rebuild complete for {project_name}: {len(documents)} files indexed")

    asyncio.get_running_loop().run_in_executor(None, do_rebuild)

    return JSONResponse({
        "status": "rebuilding",
        "project": project_name,
        "collection": collection_name,
        "folder": folder_path
    })


def sanitize_collection_name(folder_name: str) -> str:
    """Convert folder name to a valid collection name by replacing forward slashes with underscores."""
    return folder_name.replace("/", "_")


class CodeIndexerEventHandler(FileSystemEventHandler):
    def __init__(self, folder_name: str,collection_name: str):
        self.folder_name = folder_name
        self.collection_name = collection_name
        self.ignore_dirs = set(config["ignore_dirs"])
        self.ignore_files = set(config["ignore_files"])
        self.file_extensions = set(config["file_extensions"])

    def on_created(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_change(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_change(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_deletion(event.src_path)

    def _handle_file_change(self, file_path: str):
        try:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, config["projects_root"])

            # Load and process the single file
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()

            if documents:
                # Update metadata with relative path
                documents[0].metadata["file_path"] = rel_path

                # Process and index the document
                process_and_index_documents(
                    documents,
                    self.collection_name,
                    "chroma_db"
                )
                logger.info(f"Indexed updated file: {rel_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    def _handle_file_deletion(self, file_path: str):
        try:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, config["projects_root"])

            # Get the collection
            collection = chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=embedding_function
            )

            # Delete all chunks from this file
            collection.delete(
                where={"file_path": rel_path}
            )
            logger.info(f"Removed indexed chunks for deleted file: {rel_path}")
        except Exception as e:
            logger.error(f"Error removing chunks for {file_path}: {e}")


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

        # Initialize embedding function using Ollama with the bge-m3 model
        ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://host.docker.internal:11434"
        )
        embedding_function = embedding_functions.OllamaEmbeddingFunction(
            model_name="bge-m3",
            url=ollama_base_url,
        )
        logger.info(
            f"Embedding function initialized with Ollama model 'bge-m3' "
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
                    "http://host.docker.internal:11434"
                )
                embedding_function = embedding_functions.OllamaEmbeddingFunction(
                    model_name="bge-m3",
                    base_url=ollama_base_url,
                )
            except Exception as embed_err:
                logger.error(
                    f"Failed to create fallback embedding function: "
                    f"{embed_err}"
                )
        return False


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
                elif language in ["js", "jsx", "ts", "tsx"]:
                    parser_language = "javascript"
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

                        from llama_index.core.schema import TextNode
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

                    from llama_index.core.schema import TextNode
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

            if not nodes:
                logger.warning(f"No nodes generated for {file_path}")
                continue

            # Detailed per-file processing log removed to reduce noise; progress is shown on stdout.'
            # logger.info(f"Processing {file_path}: {len(nodes)} chunks")

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
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                }

                ids.append(chunk_id)
                texts.append(node.text)
                metadatas.append(metadata)

            # Add nodes to ChromaDB collection
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            total_nodes += len(nodes)

            # Progress output: "{progress}% / {total_documents}" on a single line
            progress = int((doc_index / total_documents) * 100)
            print(f"\r{progress}% / {total_documents}", end="", flush=True)

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


async def index_projects():
    """Set up file system watchers for all configured projects."""
    global observers
    try:
        for folder in config["folders_to_index"]:
            # First perform initial indexing if needed
            success = await perform_initial_indexing(folder)
            if not success:
                logger.error(f"Failed to perform initial indexing for {folder}")
                continue
            observer = Observer()
            folder_path = os.path.join(config["projects_root"], folder)
            logger.info(f"Setting up file watcher for {folder}")
            collection_name = sanitize_collection_name(folder)
            observe_folder(observer, folder_path, collection_name)
            # Create an observer and event handler for this folder
            observers.append(observer)
            observer.start()

    except Exception as e:
        logger.error(f"Error in file watching setup: {e}")
        # Clean up observers on error
        for observer in observers:
            observer.stop()
        observers.clear()


@mcp.tool(
    name="search_code",
)
async def search_code(
    query:  Annotated[ str, "Natural-language question about the codebase to search for." ],
    project: Annotated[ str, "Project or collection name to search in (typically the current workspace name, last folder name in the path)." ]    ,
    n_results: Annotated[ int, "Maximum number of matching code snippets to return." ]=8,
    threshold: Annotated[ float, "Minimum relevance score (0–100) a result must meet to be included in the response." ]=30.0,
) -> str:
    """
    Search the indexed codebase using a natural language query and return the most relevant code snippets."
    """
    try:
        logger.info(f"Running search_code with parameters: query={query}, project={project}, n_results={n_results}, threshold={threshold}");logger.info(f"Running search_code with parameters: query={query}, project={project}, n_results={n_results}, threshold={threshold}");
        if not chroma_client or not embedding_function:
            logger.error("ChromaDB client or embedding function not initialized")
            return json.dumps({
                "error": "Search system not properly initialized",
                "results": [],
                "total_results": 0
            })
        # Get all collections
        collections = chroma_client.list_collections()
        # Find matching collections
        matching_collections = []

        project_name = project.lower()
        for collection in collections:
            # The collection name might be in format "customerX_project1" or just "project1"
            # We want to match if project_name fully matches the part after the last _ (if any)
            collection_name = collection.name;
            collection_parts = collection_name.lower().split('_')
            if collection_parts[-1] == project_name:
                matching_collections.append(collection_name)

        if not matching_collections:
            logger.error(f"No collections found matching project {project}")
            return json.dumps({
                "error": f"No collections found matching project {project}",
                "results": [],
                "total_results": 0
            })

        # Search in all matching collections and combine results
        all_results = []

        for collection in matching_collections:
            collection = chroma_client.get_collection(collection)

            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            if results["documents"] and results["documents"][0]:
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    similarity = (1 - distance) * 100
                    if similarity >= threshold:
                        all_results.append({
                            "text": doc,
                            "file_path": meta.get("file_path", "Unknown file"),
                            "language": meta.get("language", "text"),
                            "start_line": int(meta.get("start_line", 0)),
                            "end_line": int(meta.get("end_line", 0)),
                            "relevance": round(similarity, 1),
                            "collection": collection.name  # Add collection name for debugging
                        })

        # Sort results by relevance
        all_results.sort(key=lambda x: x["relevance"], reverse=True)

        # Take top n_results
        final_results = all_results[:n_results]

        return json.dumps({
            "results": final_results,
            "total_results": len(final_results)
        })

    except Exception as e: 
        logger.error(
            f"Error in search_code: {str(e)}\n{traceback.format_exc()}"
        )
        return json.dumps({
            "error": str(e),
            "results": [],
            "total_results": 0
        })


def folder_contains_ignored_folders(folder_path: str) -> bool:
    """Check if a folder contains any of the ignored folders."""
    global config
    ignore_dirs = set(config["ignore_dirs"])
    for name in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, name)):
            continue
        if  name in ignore_dirs:
            return True
    return False

def observe_folder(observer: Observer, folder_path: str, collection_name: str):
    """Observe a folder for changes."""
    global config
    ignore_dirs = set(config["ignore_dirs"])
    if (folder_contains_ignored_folders(folder_path)):
        for name in os.listdir(folder_path):
            if name in ignore_dirs:
                continue
            if os.path.isdir(os.path.join(folder_path, name)):
                observe_folder(observer, os.path.join(folder_path, name), collection_name)
            else:
                observe_file(observer, os.path.join(folder_path, name), collection_name)    
    else:
        event_handler = CodeIndexerEventHandler(folder_path, collection_name)
        observer.schedule(event_handler, folder_path, recursive=True)
    
    
def observe_file(observer: Observer, file_path: str, collection_name: str):
    """Observe a file for changes."""
    global config
    ignore_dirs = set(config["ignore_dirs"])
    ignore_files = set(config["ignore_files"])
    file_extensions = set(config["file_extensions"])
    if not is_valid_file(file_path, ignore_dirs, file_extensions, ignore_files):
        return
    event_handler = CodeIndexerEventHandler(file_path, collection_name)
    observer.schedule(event_handler, file_path, recursive=False)
    logger.info(f"Started watching file {file_path}")

# Run initialization before starting MCP
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
