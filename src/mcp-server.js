import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { MilvusClient, DataType, FunctionType } from '@zilliz/milvus2-sdk-node';
import { embedText, EMBEDDING_DIMENSIONS } from 'polytokenizer';

const DEFAULT_EMBEDDING_MODEL = 'google/text-embedding-004';

class MilvusMCPServer {
  constructor(host, port, fixedCollection, embeddingModel) {
    this.host = host;
    this.port = port;
    this.fixedCollection = fixedCollection;
    this.embeddingModel = embeddingModel || DEFAULT_EMBEDDING_MODEL;

    if (!host || !port) {
      throw new Error('Milvus host and port are required');
    }

    this.client = new MilvusClient({
      address: `${host}:${port}`,
    });

    this.server = new Server(
      {
        name: 'milvus-mcp-server',
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

  async ensureCollection(collectionName) {
    if (!collectionName) {
      throw new Error('Collection name is required either as argument or default');
    }

    const hasCollection = await this.client.hasCollection({
      collection_name: collectionName,
    });

    if (!hasCollection.value) {
      const expectedDim = EMBEDDING_DIMENSIONS[this.embeddingModel];

      if (!expectedDim) {
        const availableModels = Object.keys(EMBEDDING_DIMENSIONS);
        throw new Error(`Model '${this.embeddingModel}' not found in EMBEDDING_DIMENSIONS. Available models: ${availableModels.join(', ')}`);
      }

      // Create schema for collection with both semantic and fulltext search capabilities
      const schema = [
        {
          name: 'id',
          data_type: DataType.VarChar,
          max_length: 128,
          is_primary_key: true,
        },
        {
          name: 'content',
          data_type: DataType.VarChar,
          max_length: 65535,
          enable_analyzer: true,  // Enable text analysis for fulltext search
          enable_match: true,     // Enable text matching
        },
        {
          name: 'embedding',
          data_type: DataType.FloatVector,
          dim: expectedDim,
        },
        {
          name: 'sparse',
          data_type: DataType.SparseFloatVector,  // Sparse vector for BM25 fulltext search
        },
      ];

      // Define BM25 function for fulltext search
      const functions = [
        {
          name: 'content_bm25_emb',
          description: 'BM25 function for fulltext search',
          type: FunctionType.BM25,
          input_field_names: ['content'],
          output_field_names: ['sparse'],
          params: {},
        },
      ];

      // Index parameters for both semantic and fulltext search
      const index_params = [
        {
          field_name: 'embedding',
          index_type: 'FLAT',
          metric_type: 'L2',
        },
        {
          field_name: 'sparse',
          index_type: 'SPARSE_INVERTED_INDEX',
          metric_type: 'BM25',
        },
      ];

      // Create collection with schema, functions, and index parameters
      await this.client.createCollection({
        collection_name: collectionName,
        schema: schema,
        functions: functions,
        index_params: index_params,
        enable_dynamic_field: true,
      });
    }

    await this.client.loadCollection({
      collection_name: collectionName,
    });
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
      const collectionName = this.getCollectionName(args);
      await this.ensureCollection(collectionName);

      const embedding = await this.generateEmbedding(args.content);
      const generatedId = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const data = [{
        id: generatedId,
        content: args.content,  // Content will automatically generate sparse vector via BM25 function
        embedding,   // Dense vector for semantic search
        metadata_json: JSON.stringify(args.metadata || {}),
        created_at: new Date().toISOString(),
      }];

      await this.client.insert({
        collection_name: collectionName,
        data: data,
      });

      await this.client.flush({
        collection_names: [collectionName],
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: true,
              operation: 'store',
              result: {
                id: generatedId,
                collection: collectionName,
                content_length: args.content.length,
                embedding_dimensions: embedding.length,
                embedding_model: this.embeddingModel,
                metadata: args.metadata || {},
                created_at: new Date().toISOString()
              }
            }, null, 2)
          }
        ]
      };
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
      const collectionName = this.getCollectionName(args);
      await this.ensureCollection(collectionName);

      const mode = args.mode || 'semantic';
      const limit = args.limit || 10;
      const outputFields = ['id', 'content', 'metadata_json', 'created_at'];
      let results;

      if (mode === 'semantic') {
        // Semantic search using dense vectors
        const queryEmbedding = await this.generateEmbedding(args.query);

        results = await this.client.search({
          collection_name: collectionName,
          data: [queryEmbedding],
          anns_field: 'embedding',
          limit: limit,
          params: { nprobe: 10 },
          output_fields: outputFields,
        });
      } else if (mode === 'fulltext') {
        // Fulltext search using BM25 sparse vectors
        results = await this.client.search({
          collection_name: collectionName,
          data: [args.query],  // Raw text query - Milvus will handle BM25 conversion
          anns_field: 'sparse',
          limit: limit,
          params: {
            drop_ratio_search: 0.2
          },
          output_fields: outputFields,
        });
      } else {
        throw new Error(`Unknown search mode: ${mode}`);
      }

      const memories = [];

      // results.results is always an array for search operations
      for (const item of results.results) {
        const { id, content, metadata_json, created_at, distance, score } = item;
        const similarity = distance !== undefined ?
          1.0 / (1.0 + distance) :
          (score || 1.0);

        let metadata = {};
        try {
          metadata = JSON.parse(metadata_json || '{}');
        } catch (e) {
          metadata = {};
        }

        memories.push({
          id: id,
          content: content,
          similarity: similarity,
          metadata,
          created_at,
        });
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
      const collectionName = this.getCollectionName(args);

      // First validate the ID format
      if (!args.id || !args.id.match(/^mem_\d+_[a-z0-9]+$/)) {
        throw new Error(`Invalid memory ID format. Expected format: mem_timestamp_randomstring, got: ${args.id}`);
      }

      // Check if collection exists
      const hasCollection = await this.client.hasCollection({ collection_name: collectionName });
      if (!hasCollection.value) {
        throw new Error(`Collection "${collectionName}" does not exist`);
      }

      await this.ensureCollection(collectionName);

      // Check if the memory exists before attempting deletion
      const queryResult = await this.client.query({
        collection_name: collectionName,
        filter: `id == "${args.id}"`,
        output_fields: ['id'],
        limit: 1
      });

      // If no results found, the memory doesn't exist
      if (!queryResult.data || queryResult.data.length === 0) {
        throw new Error(`Memory with ID "${args.id}" not found in collection "${collectionName}"`);
      }

      // Now delete the memory since we confirmed it exists
      await this.client.delete({
        collection_name: collectionName,
        filter: `id == "${args.id}"`,
      });

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              success: true,
              operation: 'delete',
              result: {
                id: args.id,
                collection: collectionName
              }
            }, null, 2)
          }
        ]
      };
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
      const status = await this.client.checkHealth();
      if (!status.isHealthy) {
        throw new Error(`Milvus server at ${this.host}:${this.port} is not healthy: ${JSON.stringify(status)}`);
      }

      const transport = new StdioServerTransport();
      await this.server.connect(transport);
    } catch (error) {
      console.error(`Failed to connect to Milvus: ${error.toString()}`);
      process.exit(1);
    }
  }
}

export { MilvusMCPServer, DEFAULT_EMBEDDING_MODEL };
