# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based semantic code search server for Cursor IDE. Indexes codebases locally using ChromaDB vector database and Ollama embeddings (`qwen3-embedding:0.6b`), exposed via an MCP (Model Context Protocol) server on port 8978. Provides real-time file watching for incremental index updates.

## Commands

### Run the server
```bash
docker-compose up -d
# or
bash start.sh
```

### Rebuild and restart
```bash
bash restart.sh
```

### Run all tests
```bash
source .venv/bin/activate && pytest tests/ -v
# or
bash run_tests.sh
```
Tests require Ollama running locally with the `qwen3-embedding:0.6b` model.

### Run a single test
```bash
source .venv/bin/activate && pytest tests/test_integration.py::TestPythonFileIndexing::test_name -v
```

### Test MCP endpoint manually
```bash
bash test_mcp.sh <project-name> "<search query>"
```

### Force re-index a project
```bash
bash reset.sh <project-name>
```

### Build Docker image
```bash
docker-compose build
```

## Architecture

**Entry point:** `src/main.py` — initializes ChromaDB + Ollama embedding, starts file watcher, runs FastMCP HTTP server.

**Three core modules:**

- **`src/code_indexer.py`** — Document loading, chunking (LlamaIndex CodeSplitter with tree-sitter), comment reattachment logic, and ChromaDB storage. The `reattach_leading_comments()` function ensures comments stay attached to the function/class they describe across chunk boundaries.

- **`src/mcp.py`** — FastMCP server exposing `search_code` tool (natural-language query → ChromaDB vector similarity search → ranked results) and `/rebuild/{project_name}` admin endpoint.

- **`src/observer.py`** — Watchdog-based file system watcher that incrementally updates the ChromaDB index on file create/modify/delete events.

**Data flow:** Config from `.env` → load files → tree-sitter chunking → Ollama embeddings → ChromaDB storage. Queries go through FastMCP → ChromaDB vector search → results with file path, language, line numbers, relevance scores.

## Key Implementation Details

- Code chunking: 40-line chunks, 15-line overlap, 1500 char max, language-specific tree-sitter parsers with line-based fallback
- Comment reattachment handles `#`, `//`, `/* */`, `/** */` styles
- Supported languages: Python, JS/TS/JSX/TSX, Java, C/C++, C#, Go, Ruby, PHP, Swift, Kotlin, Rust, Scala, plus text/markdown/config formats
- Configuration via `.env` file (see `.env.example` for template)
- ChromaDB persisted to `chroma_db/` directory
- Default ignore patterns exclude `node_modules`, `__pycache__`, `.git`, lock files, etc.
