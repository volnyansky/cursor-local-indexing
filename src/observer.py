import os
import logging
import asyncio

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from llama_index.core import SimpleDirectoryReader

import code_indexer

logger = logging.getLogger(__name__)

observers = []


class CodeIndexerEventHandler(FileSystemEventHandler):
    def __init__(self, folder_name: str, collection_name: str):
        self.folder_name = folder_name
        self.collection_name = collection_name
        self.ignore_dirs = set(code_indexer.config["ignore_dirs"])
        self.ignore_files = set(code_indexer.config["ignore_files"])
        self.file_extensions = set(code_indexer.config["file_extensions"])

    def on_created(self, event):
        if event.is_directory:
            return
        if code_indexer.is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_change(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if code_indexer.is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_change(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        if code_indexer.is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._handle_file_deletion(event.src_path)

    def _handle_file_change(self, file_path: str):
        try:
            # Calculate relative path
            rel_path = os.path.relpath(file_path, code_indexer.config["projects_root"])

            # Load and process the single file
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()

            if documents:
                # Update metadata with relative path
                documents[0].metadata["file_path"] = rel_path

                # Process and index the document
                code_indexer.process_and_index_documents(
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
            rel_path = os.path.relpath(file_path, code_indexer.config["projects_root"])

            # Get the collection
            collection = code_indexer.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=code_indexer.embedding_function
            )

            # Delete all chunks from this file
            collection.delete(
                where={"file_path": rel_path}
            )
            logger.info(f"Removed indexed chunks for deleted file: {rel_path}")
        except Exception as e:
            logger.error(f"Error removing chunks for {file_path}: {e}")


def folder_contains_ignored_folders(folder_path: str) -> bool:
    """Check if a folder contains any of the ignored folders."""
    ignore_dirs = set(code_indexer.config["ignore_dirs"])
    for name in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, name)):
            continue
        if name in ignore_dirs:
            return True
    return False


def observe_folder(observer: Observer, folder_path: str, collection_name: str):
    """Observe a folder for changes."""
    ignore_dirs = set(code_indexer.config["ignore_dirs"])
    if folder_contains_ignored_folders(folder_path):
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
    ignore_dirs = set(code_indexer.config["ignore_dirs"])
    ignore_files = set(code_indexer.config["ignore_files"])
    file_extensions = set(code_indexer.config["file_extensions"])
    if not code_indexer.is_valid_file(file_path, ignore_dirs, file_extensions, ignore_files):
        return
    event_handler = CodeIndexerEventHandler(file_path, collection_name)
    observer.schedule(event_handler, file_path, recursive=False)
    logger.info(f"Started watching file {file_path}")


async def index_projects():
    """Set up file system watchers for all configured projects."""
    global observers
    try:
        for folder in code_indexer.config["folders_to_index"]:
            # First perform initial indexing if needed
            success = await code_indexer.perform_initial_indexing(folder)
            if not success:
                logger.error(f"Failed to perform initial indexing for {folder}")
                continue
            observer = Observer()
            folder_path = os.path.join(code_indexer.config["projects_root"], folder)
            logger.info(f"Setting up file watcher for {folder}")
            collection_name = code_indexer.sanitize_collection_name(folder)
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
