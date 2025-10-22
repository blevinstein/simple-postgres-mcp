import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import pg from 'pg';
import { embedText, EMBEDDING_DIMENSIONS } from 'polytokenizer';

const DEFAULT_EMBEDDING_MODEL = 'google/text-embedding-004';

class PostgresMCPServer {
  constructor(host, port, database, fixedCollection, embeddingModel) {
    this.host = host;
    this.port = port;
    this.database = database;
    this.fixedCollection = fixedCollection;
    this.embeddingModel = embeddingModel || DEFAULT_EMBEDDING_MODEL;

    if (!host || !port || !database) {
      throw new Error('PostgreSQL host, port, and database are required');
    }

    this.pool = new pg.Pool({
      host: host,
      port: parseInt(port),
      database: database,
      user: process.env.PGUSER || 'postgres',
      password: process.env.PGPASSWORD || 'postgres',
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    this.server = new Server(
      {
        name: 'postgres-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const hasFixedCollection = !!this.fixedCollection;

      return {
        tools: [
          {
            name: 'store_memory',
            description: 'Store a memory/document in vector storage, to be indexed for semantic and fulltext search. The collection will be created if it does not exist.',
            inputSchema: {
              type: 'object',
              properties: {
                content: {
                  type: 'string',
                  description: 'The text content to store'
                },
                metadata: {
                  type: 'object',
                  description: 'Additional metadata to store with the memory',
                  additionalProperties: true
                },
                ...(hasFixedCollection ? {} : {
                  collection: {
                    type: 'string',
                    description: 'Collection name'
                  }
                }),
              },
              required: ['content']
            }
          },
          {
            name: 'search_memory',
            description: 'Search for memories/documents in vector storage',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Search query text'
                },
                mode: {
                  type: 'string',
                  enum: ['semantic', 'fulltext'],
                  default: 'semantic',
                  description: 'Search mode: semantic (embedding-based) or fulltext'
                },
                limit: {
                  type: 'number',
                  default: 10,
                  description: 'Maximum number of results to return'
                },
                ...(hasFixedCollection ? {} : {
                  collection: {
                    type: 'string',
                    description: 'Collection name'
                  }
                }),
              },
              required: ['query']
            }
          },
          {
            name: 'forget_memory',
            description: 'Delete a memory/document from vector storage',
            inputSchema: {
              type: 'object',
              properties: {
                id: {
                  type: 'string',
                  description: 'Auto-generated ID of the memory to delete'
                },
                ...(hasFixedCollection ? {} : {
                  collection: {
                    type: 'string',
                    description: 'Collection name'
                  }
                }),
              },
              required: ['id']
            }
          }
        ]
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'store_memory':
          return await this.storeMemory(args);
        case 'search_memory':
          return await this.searchMemory(args);
        case 'forget_memory':
          return await this.forgetMemory(args);
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  getCollectionName(args) {
    return args.collection || this.fixedCollection;
  }

  async ensureTable(tableName) {
    if (!tableName) {
      throw new Error('Table name is required either as argument or default');
    }

    // Validate table name to prevent SQL injection
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(tableName)) {
      throw new Error('Invalid table name. Must start with letter/underscore and contain only alphanumeric characters and underscores');
    }

    const expectedDim = EMBEDDING_DIMENSIONS[this.embeddingModel];

    if (!expectedDim) {
      const availableModels = Object.keys(EMBEDDING_DIMENSIONS);
      throw new Error(`Model '${this.embeddingModel}' not found in EMBEDDING_DIMENSIONS. Available models: ${availableModels.join(', ')}`);
    }

    const client = await this.pool.connect();
    try {
      // Check if table exists
      const checkTableQuery = `
        SELECT EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_name = $1
        );
      `;
      const tableExists = await client.query(checkTableQuery, [tableName]);

      if (!tableExists.rows[0].exists) {
        // Create table with pgvector and full-text search support
        const createTableQuery = `
          CREATE TABLE ${tableName} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(${expectedDim}),
            content_fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            metadata_json JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
          );
        `;
        await client.query(createTableQuery);

        // Create GIN index for full-text search
        const createFtsIndexQuery = `
          CREATE INDEX ${tableName}_content_fts_idx ON ${tableName} USING GIN(content_fts);
        `;
        await client.query(createFtsIndexQuery);

        // Create HNSW index for vector similarity search
        const createVectorIndexQuery = `
          CREATE INDEX ${tableName}_embedding_idx ON ${tableName} USING hnsw (embedding vector_l2_ops);
        `;
        await client.query(createVectorIndexQuery);
      }
    } finally {
      client.release();
    }
  }

  async generateEmbedding(text) {
    try {
      const result = await embedText(this.embeddingModel, text);
      return result.vector;
    } catch (error) {
      if (error.message.includes('API key not found')) {
        throw new Error(`API key not configured for embedding model ${this.embeddingModel}. Please set the appropriate environment variable (e.g., GEMINI_API_KEY for Google models, OPENAI_API_KEY for OpenAI models). Original error: ${error.message}`);
      }
      throw new Error(`Failed to generate embedding with ${this.embeddingModel}: ${error.message}`);
    }
  }

  async storeMemory(args) {
    try {
      const tableName = this.getCollectionName(args);
      await this.ensureTable(tableName);

      const embedding = await this.generateEmbedding(args.content);
      const generatedId = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const insertQuery = `
        INSERT INTO ${tableName} (id, content, embedding, metadata_json)
        VALUES ($1, $2, $3, $4)
        RETURNING created_at;
      `;

      const client = await this.pool.connect();
      try {
        const result = await client.query(insertQuery, [
          generatedId,
          args.content,
          JSON.stringify(embedding),
          JSON.stringify(args.metadata || {})
        ]);

        const createdAt = result.rows[0].created_at;

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                success: true,
                operation: 'store',
                result: {
                  id: generatedId,
                  collection: tableName,
                  content_length: args.content.length,
                  embedding_dimensions: embedding.length,
                  embedding_model: this.embeddingModel,
                  metadata: args.metadata || {},
                  created_at: createdAt
                }
              }, null, 2)
            }
          ]
        };
      } finally {
        client.release();
      }
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: false,
              operation: 'store',
              error: error.message
            }, null, 2)
          }
        ],
        isError: true
      };
    }
  }

  async searchMemory(args) {
    try {
      const tableName = this.getCollectionName(args);
      await this.ensureTable(tableName);

      const mode = args.mode || 'semantic';
      const limit = args.limit || 10;
      const memories = [];

      const client = await this.pool.connect();
      try {
        let result;

        if (mode === 'semantic') {
          // Semantic search using pgvector
          const queryEmbedding = await this.generateEmbedding(args.query);

          const searchQuery = `
            SELECT id, content, metadata_json, created_at, embedding <-> $1 AS distance
            FROM ${tableName}
            ORDER BY embedding <-> $1
            LIMIT $2;
          `;

          result = await client.query(searchQuery, [
            JSON.stringify(queryEmbedding),
            limit
          ]);

          // Convert distance to similarity score
          for (const row of result.rows) {
            const similarity = 1.0 / (1.0 + row.distance);
            let metadata = {};
            try {
              metadata = typeof row.metadata_json === 'string'
                ? JSON.parse(row.metadata_json)
                : (row.metadata_json || {});
            } catch (e) {
              metadata = {};
            }

            memories.push({
              id: row.id,
              content: row.content,
              similarity: similarity,
              metadata,
              created_at: row.created_at,
            });
          }
        } else if (mode === 'fulltext') {
          // Full-text search using PostgreSQL tsvector
          const searchQuery = `
            SELECT id, content, metadata_json, created_at,
                   ts_rank(content_fts, to_tsquery('english', $1)) AS rank
            FROM ${tableName}
            WHERE content_fts @@ to_tsquery('english', $1)
            ORDER BY rank DESC
            LIMIT $2;
          `;

          // Convert query to tsquery format (replace spaces with &)
          const tsQuery = args.query.trim().split(/\s+/).join(' & ');

          result = await client.query(searchQuery, [tsQuery, limit]);

          for (const row of result.rows) {
            let metadata = {};
            try {
              metadata = typeof row.metadata_json === 'string'
                ? JSON.parse(row.metadata_json)
                : (row.metadata_json || {});
            } catch (e) {
              metadata = {};
            }

            memories.push({
              id: row.id,
              content: row.content,
              similarity: row.rank,
              metadata,
              created_at: row.created_at,
            });
          }
        } else {
          throw new Error(`Unknown search mode: ${mode}`);
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                success: true,
                operation: 'search',
                result: {
                  query: args.query,
                  mode: mode,
                  count: memories.length,
                  memories: memories
                }
              }, null, 2)
            }
          ]
        };
      } finally {
        client.release();
      }
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: false,
              operation: 'search',
              error: error.message
            }, null, 2)
          }
        ],
        isError: true
      };
    }
  }

  async forgetMemory(args) {
    try {
      const tableName = this.getCollectionName(args);

      // First validate the ID format
      if (!args.id || !args.id.match(/^mem_\d+_[a-z0-9]+$/)) {
        throw new Error(`Invalid memory ID format. Expected format: mem_timestamp_randomstring, got: ${args.id}`);
      }

      const client = await this.pool.connect();
      try {
        // Check if table exists
        const checkTableQuery = `
          SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = $1
          );
        `;
        const tableExists = await client.query(checkTableQuery, [tableName]);

        if (!tableExists.rows[0].exists) {
          throw new Error(`Collection "${tableName}" does not exist`);
        }

        // Check if the memory exists before attempting deletion
        const checkMemoryQuery = `
          SELECT id FROM ${tableName} WHERE id = $1;
        `;
        const memoryExists = await client.query(checkMemoryQuery, [args.id]);

        if (memoryExists.rows.length === 0) {
          throw new Error(`Memory with ID "${args.id}" not found in collection "${tableName}"`);
        }

        // Delete the memory
        const deleteQuery = `
          DELETE FROM ${tableName} WHERE id = $1;
        `;
        await client.query(deleteQuery, [args.id]);

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                success: true,
                operation: 'delete',
                result: {
                  id: args.id,
                  collection: tableName
                }
              }, null, 2)
            }
          ]
        };
      } finally {
        client.release();
      }
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: false,
              operation: 'delete',
              error: error.message
            }, null, 2)
          }
        ],
        isError: true
      };
    }
  }

  async run() {
    try {
      // Test PostgreSQL connection
      const client = await this.pool.connect();
      try {
        await client.query('SELECT 1');
        // Ensure pgvector extension is installed
        await client.query('CREATE EXTENSION IF NOT EXISTS vector');
      } finally {
        client.release();
      }

      const transport = new StdioServerTransport();
      await this.server.connect(transport);
    } catch (error) {
      console.error(`Failed to connect to PostgreSQL: ${error.toString()}`);
      process.exit(1);
    }
  }

  async close() {
    await this.pool.end();
  }
}

export { PostgresMCPServer, DEFAULT_EMBEDDING_MODEL };
