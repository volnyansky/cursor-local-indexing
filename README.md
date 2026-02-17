# Local Code Indexing for Cursor

An experimental Python-based server that **locally** indexes codebases using ChromaDB and provides a semantic search tool via an MCP (Model Context Protocol) server for tools like Cursor.

## Setup

1. Clone and enter the repository:
   ```bash
   git clone <repository-url>
   cd cursor-local-indexing
   ```

2. Create a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Configure your `.env` file:
   ```env
   PROJECTS_ROOT=~/your/projects/root    # Path to your projects directory
   FOLDERS_TO_INDEX=project1,project2    # Comma-separated list of folders to index
   ```

   Example:
   ```env
   PROJECTS_ROOT=~/projects
   FOLDERS_TO_INDEX=project1,project2
   ```
4. Install Ollama, pull bge-m3 model

5. Start the indexing server:
   ```bash
   docker-compose up -d
   ```

6. Configure Cursor to use the local search server:
   Create or edit `~/.cursor/mcp.json`:
   ```json
   {
     "mcpServers": {
       "workspace-code-search": {
         "url": "http://localhost:8978/sse"
       }
     }
   }
   ```

7. Restart Cursor IDE to apply the changes.

The server will start indexing your specified projects, and you'll be able to use semantic code search within Cursor when those projects are active.

8. Open a project that you configured as indexed.

add following into rules section of cursor settings
[[calls]]
match = "For any request which is related to the code base (finding implementation, understanding behavior, debugging, refactoring, or extending existing code). Use this tool *before* Grep, SemanticSearch, or reading specific files."
tool = "search_code"
```

8. Start using the Cursor Agent mode and see it doing local vector searches!