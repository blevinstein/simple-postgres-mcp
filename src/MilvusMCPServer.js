const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { CallToolRequestSchema, ListToolsRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const { MilvusClient, DataType } = require('@zilliz/milvus2-sdk-node');

const DEFAULT_EMBEDDING_MODEL = 'google/text-embedding-004';

class MilvusMCPServer {
  constructor(host, port, fixedCollection, embeddingModel) {
    this.host = host;
    this.port = port;
    this.fixedCollection = fixedCollection;
    this.embeddingModel = embeddingModel || DEFAULT_EMBEDDING_MODEL;
    this.embedText = null;
    
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

  // TODO Solve build issues that require CommonJS in this file, and therefore this hacky ES6 import
  async loadPolytokenizer() {
    if (!this.embedText) {
      const polytokenizer = await import('polytokenizer');
      this.embedText = polytokenizer.embedText;
      this.EMBEDDING_DIMENSIONS = polytokenizer.EMBEDDING_DIMENSIONS;
    }
    return { embedText: this.embedText, EMBEDDING_DIMENSIONS: this.EMBEDDING_DIMENSIONS };
  }

  setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const hasFixedCollection = !!this.fixedCollection;
      
      return {
        tools: [
          {
            name: 'store_memory',
            description: 'Store a memory/document in Milvus with automatic embedding generation',
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
            description: 'Search for memories/documents in Milvus',
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
            description: 'Delete a memory/document from Milvus',
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
      const { EMBEDDING_DIMENSIONS } = await this.loadPolytokenizer();
      const expectedDim = EMBEDDING_DIMENSIONS[this.embeddingModel];
      
      if (!expectedDim) {
        const availableModels = Object.keys(EMBEDDING_DIMENSIONS);
        throw new Error(`Model '${this.embeddingModel}' not found in EMBEDDING_DIMENSIONS. Available models: ${availableModels.join(', ')}`);
      }

      await this.client.createCollection({
        collection_name: collectionName,
        fields: [
          {
            name: 'id',
            data_type: 'VarChar',
            max_length: 128,
            is_primary_key: true,
          },
          {
            name: 'embedding',
            data_type: 'FloatVector',
            dim: expectedDim,
          },
        ],
        enable_dynamic_field: true,
      });
      
      await this.client.createIndex({
        collection_name: collectionName,
        field_name: 'embedding',
        index_type: 'FLAT',
        metric_type: 'L2',
      });
    }

    await this.client.loadCollection({
      collection_name: collectionName,
    });
  }

  async generateEmbedding(text) {
    try {
      const { embedText } = await this.loadPolytokenizer();
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
        embedding: embedding,
        content: args.content,
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
            text: `Successfully stored memory in collection: ${collectionName}
- Generated ID: ${generatedId}
- Content length: ${args.content.length} characters
- Embedding dimensions: ${embedding.length}
- Embedding model: ${this.embeddingModel}
- Metadata fields: ${Object.keys(args.metadata || {}).join(', ') || 'none'}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error storing memory: ${error.message}`
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

      await this.client.flush({
        collection_names: [collectionName],
      });

      const mode = args.mode || 'semantic';
      const limit = args.limit || 10;
      const outputFields = ['id', 'content', 'metadata_json', 'created_at'];
      let results;

      if (mode === 'semantic') {
        const queryEmbedding = await this.generateEmbedding(args.query);
        
        results = await this.client.search({
          collection_name: collectionName,
          vectors: [queryEmbedding],
          search_params: {
            anns_field: 'embedding',
            topk: limit,
            metric_type: 'L2',
            params: JSON.stringify({ nprobe: 10 }),
          },
          output_fields: outputFields,
        });
      } else if (mode === 'fulltext') {
        throw new Error('Fulltext search is not currently implemented. Please use semantic search mode instead. Fulltext search requires a different schema with sparse vectors and BM25 functions.');
      } else {
        throw new Error(`Unknown search mode: ${mode}`);
      }

      const memories = [];
      const resultData = results.results || results.data || [];
      
      for (const item of resultData) {
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
          created_at
        });
      }

      return {
        content: [
          {
            type: 'text',
            text: `Found ${memories.length} memories using ${mode} search:\n\n` +
                  memories.map(m => `ID: ${m.id}\nSimilarity: ${m.similarity.toFixed(3)}\nContent: ${m.content}\n`).join('\n---\n')
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error searching memories: ${error.message}`
          }
        ],
        isError: true
      };
    }
  }

  async forgetMemory(args) {
    try {
      const collectionName = this.getCollectionName(args);
      await this.ensureCollection(collectionName);

      await this.client.delete({
        collection_name: collectionName,
        filter: `id == "${args.id}"`,
      });

      return {
        content: [
          {
            type: 'text',
            text: `Successfully deleted memory with ID: ${args.id} from collection: ${collectionName}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error deleting memory: ${error.message}`
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
        throw new Error(`Milvus server at ${this.host}:${this.port} is not healthy`);
      }
      
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
    } catch (error) {
      console.error(`Failed to connect to Milvus: ${error.message}`);
      process.exit(1);
    }
  }
}

module.exports = { MilvusMCPServer, DEFAULT_EMBEDDING_MODEL }; 