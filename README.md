# Simple Milvus MCP

A Model Context Protocol (MCP) server for [Milvus](https://milvus.io/) vector database that provides semantic and full-text search capabilities.

## Features

- **Semantic Search**: Vector-based similarity search using embeddings
- **Full-text Search**: Traditional text matching search
- **Memory Management**: Store, search, and delete documents/memories
- **Flexible Embedding Models**: Support for OpenAI, Vertex AI, and Google embedding models
- **Configurable Collections**: Use default collection or specify per-operation

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
npx simple-milvus-mcp --collection my_memories --embedding-model openai/text-embedding-3-small

# Development mode (without building)
pnpm run dev --help
```

### Command Line Options

- `--host`: Milvus server host (default: `localhost`)
- `--port`: Milvus server port (default: `19530`)
- `--collection`: Default collection name (optional)
- `--embedding-model`: Default embedding model (default: `openai/text-embedding-3-small`)

### Available Tools

#### 1. `store_memory`
Store a document/memory in Milvus with automatic embedding generation.

**Parameters:**
- `id` (string, required): Unique identifier for the memory
- `content` (string, required): The text content to store
- `metadata` (object, optional): Additional metadata to store
- `collection` (string, optional): Collection name (if not set as default)
- `embedding_model` (string, optional): Embedding model to use

#### 2. `search_memory`
Search for memories/documents in Milvus.

**Parameters:**
- `query` (string, required): Search query text
- `mode` (string, optional): Search mode - `semantic` or `fulltext` (default: `semantic`)
- `limit` (number, optional): Maximum number of results (default: 10)
- `collection` (string, optional): Collection name (if not set as default)
- `embedding_model` (string, optional): Embedding model for semantic search

#### 3. `forget_memory`
Delete a memory/document from Milvus.

**Parameters:**
- `id` (string, required): Unique identifier of the memory to delete
- `collection` (string, optional): Collection name (if not set as default)
