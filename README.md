# Simple Milvus MCP

A Model Context Protocol (MCP) server for [Milvus](https://milvus.io/) vector database that provides semantic and full-text search capabilities using both dense embeddings and BM25 sparse vectors.

## Features

- **Semantic Search**: Vector-based similarity search using dense embeddings
- **Full-text Search**: BM25-based keyword search using sparse vectors
- **Memory Management**: Store, search, and delete documents/memories with auto-generated IDs
- **Flexible Embedding Models**: Support for OpenAI, Vertex AI, and Google embedding models
- **Automatic Schema Management**: Collections are created automatically with proper BM25 configuration
- **Configurable Collections**: Use default collection or specify per-operation

## Prerequisites

- **Milvus Server**: Running Milvus 2.5+ instance
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

## Usage

### Running the MCP Server

```bash
# Using npx (after building)
npx simple-milvus-mcp --host localhost --port 19530

# With a default collection
npx simple-milvus-mcp --collection my_memories

# Using OpenAI embeddings
npx simple-milvus-mcp --embedding-model openai/text-embedding-3-small

# Development mode (without building)
pnpm run dev --help
```

### Command Line Options

- `--host`: Milvus server host (default: `localhost`)
- `--port`: Milvus server port (default: `19530`)
- `--collection`: Default collection name (optional - collections created as needed)
- `--embedding-model`: Embedding model to use (default: `google/text-embedding-004`)

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
Delete a memory/document from Milvus using its auto-generated ID.

**Parameters:**
- `id` (string, required): Auto-generated ID of the memory to delete (format: `mem_timestamp_randomstring`)
- `collection` (string, optional): Collection name (if not set as default)

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
