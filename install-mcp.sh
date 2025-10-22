#!/bin/bash
# Install simple-postgres-mcp as an MCP server for Claude Code

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

claude mcp add postgres-memories \
  --transport stdio \
  --env PGUSER=postgres \
  --env PGPASSWORD=postgres \
  --env "GEMINI_API_KEY=$GEMINI_API_KEY" \
  -- node \
  "$SCRIPT_DIR/src/index.js" \
  --host "localhost" \
  --port "5432" \
  --database "mcp_memories" \
  --collection "my_memories" \
  --embedding-model "google/text-embedding-004"

echo "✓ MCP server installed. Restart Claude Code to use it."
