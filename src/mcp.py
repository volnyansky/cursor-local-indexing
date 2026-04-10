import os
import json
import traceback
import asyncio
import logging

from typing_extensions import Annotated
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

import src.code_indexer as code_indexer

logger = logging.getLogger(__name__)

mcp = FastMCP(name="Code Indexer Server")


@mcp.custom_route("/rebuild/{project_name}", methods=["POST"])
async def rebuild_index(request: Request) -> JSONResponse:
    project_name = request.path_params["project_name"]

    # Find folder matching project_name (same logic as search_code)
    matching_folder = None
    for folder in code_indexer.config["folders_to_index"]:
        collection_name = code_indexer.sanitize_collection_name(folder)
        if collection_name.lower().split("_")[-1] == project_name.lower():
            matching_folder = folder
            break

    if not matching_folder:
        return JSONResponse(
            {"error": f"Project '{project_name}' not found in configured folders"},
            status_code=404
        )

    collection_name = code_indexer.sanitize_collection_name(matching_folder)
    folder_path = os.path.join(code_indexer.config["projects_root"], matching_folder)

    def do_rebuild():
        try:
            code_indexer.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name} for rebuild")
        except Exception:
            pass  # Collection may not exist yet

        documents = code_indexer.load_documents(
            folder_path,
            ignore_dirs=set(code_indexer.config["ignore_dirs"]),
            file_extensions=set(code_indexer.config["file_extensions"]),
            ignore_files=set(code_indexer.config["ignore_files"])
        )

        if not documents:
            logger.warning(f"No indexable documents found in {folder_path}")
            return

        code_indexer.process_and_index_documents(documents, collection_name, "chroma_db")
        logger.info(f"Rebuild complete for {project_name}: {len(documents)} files indexed")

    asyncio.get_running_loop().run_in_executor(None, do_rebuild)

    return JSONResponse({
        "status": "rebuilding",
        "project": project_name,
        "collection": collection_name,
        "folder": folder_path
    })


@mcp.tool(
    name="search_code",
)
async def search_code(
    query: Annotated[str, "Natural-language question about the codebase to search for."],
    project: Annotated[str, "Project or collection name to search in (typically the current workspace name, last folder name in the path)."],
    n_results: Annotated[int, "Maximum number of matching code snippets to return."] = 8,
    threshold: Annotated[float, "Minimum relevance score (0–100) a result must meet to be included in the response."] = 30.0,
) -> str:
    """
    Search the indexed codebase using a natural language query and return the most relevant code snippets.
    """
    try:
        logger.info(f"Running search_code with parameters: query={query}, project={project}, n_results={n_results}, threshold={threshold}")
        if not code_indexer.chroma_client or not code_indexer.embedding_function:
            logger.error("ChromaDB client or embedding function not initialized")
            return json.dumps({
                "error": "Search system not properly initialized",
                "results": [],
                "total_results": 0
            })
        # Get all collections
        collections = code_indexer.chroma_client.list_collections()
        # Find matching collections
        matching_collections = []

        project_name = project.lower()
        for collection in collections:
            # The collection name might be in format "customerX_project1" or just "project1"
            # We want to match if project_name fully matches the part after the last _ (if any)
            collection_name = collection.name
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
            collection = code_indexer.chroma_client.get_collection(collection, embedding_function=code_indexer.embedding_function)

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
