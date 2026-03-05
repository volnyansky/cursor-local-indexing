#!/usr/bin/env bash
set -euo pipefail

HOST="${MCP_HOST:-http://localhost:8978}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <project> <query>"
  echo ""
  echo "  project   Project name to search in"
  echo "  query     Search query (e.g. 'pdf parsing')"
  echo ""
  echo "Example: $0 backend 'pdf parsing'"
  exit 1
fi

PROJECT="$1"
QUERY="$2"

echo "=== MCP: initialize ==="
SESSION_ID=$(curl -s -D - -X POST "$HOST/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": { "name": "test-curl", "version": "1.0" }
    }
  }' | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')

echo "Session ID: $SESSION_ID"

echo ""
echo "=== MCP: tools/list ==="
curl -s -X POST "$HOST/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }'

echo ""
echo "=== MCP: search_code (query=\"$QUERY\", project=\"$PROJECT\") ==="
curl -s -X POST "$HOST/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d "{
    \"jsonrpc\": \"2.0\",
    \"id\": 3,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"search_code\",
      \"arguments\": {
        \"query\": \"$QUERY\",
        \"project\": \"$PROJECT\"
      }
    }
  }"
