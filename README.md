# Simple PostgreSQL MCP

A Model Context Protocol (MCP) server for PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) that provides semantic and full-text search capabilities for LLM memory management.

## Why This MCP Server?

**Primary Use Case: Multi-Tenant Collection Management**
- Multiple MCP servers can use the same PostgreSQL instance with segregated storage
- Each server instance can operate on different tables (e.g., per-user, per-account, per-application)
- Enables cost-effective shared infrastructure while maintaining data isolation
- Perfect for SaaS applications where each customer needs their own vector space

**Secondary Benefit: Agent-Friendly Simplified Interface**
- Agents and LLMs can struggle with too many tool options, leading to poor decision-making
- This server provides just 3 focused tools (`store_memory`, `search_memory`, `forget_memory`)
- Simplified interface makes it easier to combine with other MCP servers and capabilities
- Optimized for memory/knowledge management workflows rather than full database administration

## Features

- **Semantic Search**: Vector-based similarity search using pgvector with L2 distance
- **Full-text Search**: PostgreSQL native tsvector/tsquery full-text search
- **Memory Management**: Store, search, and delete documents/memories with auto-generated IDs
- **Flexible Embedding Models**: Support for OpenAI, Vertex AI, and Google embedding models
- **Automatic Schema Management**: Tables are created automatically with proper indexes
- **Configurable Collections**: Use default table or specify per-operation
- **JSONB Metadata**: Rich metadata support with native PostgreSQL JSONB

## Prerequisites

- **PostgreSQL**: PostgreSQL 12+ with pgvector extension installed
- **API Keys**: Required environment variables for embedding models:
  ```bash
  # For Google models (default)
  export GEMINI_API_KEY="your_google_api_key"

  # For OpenAI models
  export OPENAI_API_KEY="your_openai_api_key"

  # For Vertex AI models
  export VERTEX_PROJECT_ID="your-gcp-project"
  export VERTEX_LOCATION="us-central1"
  export VERTEX_CREDENTIALS='{"type":"service_account","project_id":"your-project","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"...@...iam.gserviceaccount.com",...}'
  ```

## Installation

```bash
# Install dependencies
pnpm install

# Build the package
pnpm run build

# Test the installation
pnpm run test
```

## Quick Start with Docker

The easiest way to get started is using Docker Compose:

```bash
# Start PostgreSQL with pgvector
pnpm run docker:up

# View logs
pnpm run docker:logs

# Stop PostgreSQL
pnpm run docker:down
```

This starts PostgreSQL on port 5432 with:
- Database: `mcp_memories`
- User: `postgres`
- Password: `postgres`
- pgvector extension pre-installed

## Usage

### Running the MCP Server

```bash
# Using npx (after building)
npx simple-postgres-mcp --host localhost --port 5432 --database mcp_memories

# With a default collection/table
npx simple-postgres-mcp --collection my_memories

# Using OpenAI embeddings
npx simple-postgres-mcp --embedding-model openai/text-embedding-3-small

# Development mode (without building)
pnpm run dev --help
```

### Command Line Options

- `--host`: PostgreSQL server host (default: `localhost`)
- `--port`: PostgreSQL server port (default: `5432`)
- `--database`: PostgreSQL database name (default: `mcp_memories`)
- `--collection`: Default table name (optional - tables created as needed)
- `--embedding-model`: Embedding model to use (default: `google/text-embedding-004`)

### Environment Variables

- `PGUSER`: PostgreSQL username (default: `postgres`)
- `PGPASSWORD`: PostgreSQL password (default: `postgres`)

### Available Tools

#### 1. `store_memory`
Store a document/memory in Milvus with automatic embedding generation and ID creation.

**Parameters:**
- `content` (string, required): The text content to store
- `metadata` (object, optional): Additional metadata to store with the memory
- `collection` (string, optional): Collection name (if not set as default)

**Response:**
```json
{
  "success": true,
  "operation": "store",
  "result": {
    "id": "mem_1234567890_abc123def",
    "collection": "my_memories",
    "content_length": 42,
    "embedding_dimensions": 768,
    "embedding_model": "google/text-embedding-004",
    "metadata": {"topic": "AI"},
    "created_at": "2024-01-01T12:00:00.000Z"
  }
}
```

#### 2. `search_memory`
Search for memories/documents using semantic or full-text search.

**Parameters:**
- `query` (string, required): Search query text
- `mode` (string, optional): Search mode - `semantic` or `fulltext` (default: `semantic`)
- `limit` (number, optional): Maximum number of results (default: 10)
- `collection` (string, optional): Collection name (if not set as default)

**Response:**
```json
{
  "success": true,
  "operation": "search",
  "result": {
    "query": "machine learning",
    "mode": "semantic",
    "count": 2,
    "memories": [
      {
        "id": "mem_1234567890_abc123def",
        "content": "Machine learning is a subset of AI...",
        "similarity": 0.92,
        "metadata": {"topic": "AI"},
        "created_at": "2024-01-01T12:00:00.000Z"
      }
    ]
  }
}
```

#### 3. `forget_memory`
Delete a memory/document from PostgreSQL using its auto-generated ID.

**Parameters:**
- `id` (string, required): Auto-generated ID of the memory to delete (format: `mem_timestamp_randomstring`)
- `collection` (string, optional): Table name (if not set as default)

**Response:**
```json
{
  "success": true,
  "operation": "delete",
  "result": {
    "id": "mem_1234567890_abc123def",
    "collection": "my_memories"
  }
}
```

## Configuration with Claude Desktop

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "postgres-memories": {
      "command": "npx",
      "args": [
        "simple-postgres-mcp",
        "--host", "localhost",
        "--port", "5432",
        "--database", "mcp_memories",
        "--collection", "my_memories"
      ],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here",
        "PGUSER": "postgres",
        "PGPASSWORD": "postgres"
      }
    }
  }
}
```

## Architecture

### Database Schema

Each collection/table is created with the following schema:

```sql
CREATE TABLE {collection_name} (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(768),  -- pgvector, dimension varies by model
  content_fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  metadata_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX {table}_content_fts_idx ON {table} USING GIN(content_fts);
CREATE INDEX {table}_embedding_idx ON {table} USING hnsw (embedding vector_l2_ops);
```

### Search Methods

- **Semantic Search**: Uses pgvector's L2 distance operator (`<->`) for similarity
- **Full-text Search**: Uses PostgreSQL's native tsvector/tsquery with ts_rank for relevance

## License

MIT
