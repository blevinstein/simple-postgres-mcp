#!/usr/bin/env node

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { CallToolRequestSchema, ListToolsRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const yargs = require('yargs');
const { hideBin } = require('yargs/helpers');
const { MilvusClient } = require('@zilliz/milvus2-sdk-node');

const DEFAULT_EMBEDDING_MODEL = 'google/text-embedding-004';

const argv = yargs(hideBin(process.argv))
  .option('host', {
    type: 'string',
    default: 'localhost',
    description: 'Milvus server host'
  })
  .option('port', {
    type: 'string',
    default: '19530',
    description: 'Milvus server port'
  })
  .option('collection', {
    type: 'string',
    description: 'Milvus collection name (if not provided, must be specified per tool call)'
  })
  .option('embedding-model', {
    type: 'string',
    default: 'openai/text-embedding-3-small',
    description: 'Default embedding model (e.g., openai/text-embedding-3-small, vertex/text-embedding-005, google/text-embedding-004)'
  })
  .help()
  .argv;

class MilvusMCPServer {
  constructor(host, port, fixedCollection, embeddingModel) {
    this.host = host;
    this.port = port;
    this.fixedCollection = fixedCollection;
    this.embeddingModel = embeddingModel || DEFAULT_EMBEDDING_MODEL;
    this.embedText = null;
    
    // Validate that we have all required parameters
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

  async loadEmbedText() {
    if (!this.embedText) {
      // TODO Solve build issues that require CommonJS in this file, and therefore this hacky ES6 import
      const polytokenizer = await import('polytokenizer');
      this.embedText = polytokenizer.embedText;
      this.EMBEDDING_DIMENSIONS = polytokenizer.EMBEDDING_DIMENSIONS;
    }
    return this.embedText;
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
                id: {
                  type: 'string',
                  description: 'Unique identifier for the memory'
                },
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
              required: ['id', 'content']
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
                  description: 'Unique identifier of the memory to delete'
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
      const embeddingDim = this.EMBEDDING_DIMENSIONS[this.embeddingModel];
      if (!embeddingDim) {
        throw new Error(`Unknown embedding dimensions for '${this.embeddingModel}'`);
      }

      // Create collection with default schema
      await this.client.createCollection({
        collection_name: collectionName,
        fields: [
          {
            name: 'id',
            description: 'Unique identifier',
            data_type: 11, // DataType.VarChar
            is_primary_key: true,
            max_length: 100,
          },
          {
            name: 'content',
            description: 'Memory content',
            data_type: 11, // DataType.VarChar
            max_length: 65535,
          },
          {
            name: 'embedding',
            description: 'Vector embedding of content',
            data_type: 101, // DataType.FloatVector
            dim: embeddingDim,
          }
        ],
        enable_dynamic_field: true,
      });

      // Create index on the embedding field
      await this.client.createIndex({
        collection_name: collectionName,
        field_name: 'embedding',
        extra_params: {
          index_type: 'HNSW',
          metric_type: 'L2',
          params: JSON.stringify({ M: 8, efConstruction: 200 }),
        },
      });

      console.log(`Created collection '${collectionName}' with embedding dimension ${embeddingDim} for model '${embeddingModel}'`);
    }

    await this.client.loadCollection({
      collection_name: collectionName,
    });
  }

  async generateEmbedding(text) {
    try {
      const embedText = await this.loadEmbedText();
      const result = await embedText(this.embeddingModel, text);
      return result.vector;
    } catch (error) {
      throw new Error(`Failed to generate embedding with ${embeddingModel}: ${error.message}`);
    }
  }

  async storeMemory(args) {
    try {
      const collectionName = this.getCollectionName(args);
      
      await this.ensureCollection(collectionName);

      const embedding = await this.generateEmbedding(args.content);
      
      const insertData = [{
        id: args.id,
        content: args.content,
        embedding: embedding,
        ...args.metadata
      }];

      const result = await this.client.insert({
        collection_name: collectionName,
        data: insertData,
      });

      await this.client.flush({
        collection_names: [collectionName],
      });

      return {
        content: [
          {
            type: 'text',
            text: `Successfully stored memory with ID: ${args.id} in collection: ${collectionName} using embedding model: ${embeddingModel}`
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

      const mode = args.mode || 'semantic';
      const limit = args.limit || 10;
      let results;

      // Get collection fields to extract all metadata
      const collectionInfo = await this.client.describeCollection({
        collection_name: collectionName,
      });
      
      const allFields = collectionInfo.schema.fields.map(field => field.name);
      const outputFields = ['id', 'content', ...allFields.filter(f => 
        f !== 'id' && f !== 'content' && f !== 'embedding')];

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
        const escapedQuery = args.query.replace(/"/g, '\\"');
        results = await this.client.query({
          collection_name: collectionName,
          expr: `content like "%${escapedQuery}%"`,
          output_fields: outputFields,
          limit: limit,
        });
      } else {
        throw new Error(`Unknown search mode: ${mode}`);
      }

      const memories = [];
      if (results.results && results.results.length > 0) {
        for (const hit of results.results) {
          const { id, content, embedding, distance, score, ...metadata } = hit;
          const similarity = distance !== undefined ? 
            1.0 / (1.0 + distance) : 
            (score || 1.0);
          
          memories.push({
            id: id,
            content: content,
            similarity: similarity,
            metadata,
          });
        }
      } else if (results.data) {
        for (const item of results.data) {
          const { id, content, embedding, distance, score, ...metadata } = item;
          const similarity = distance !== undefined ? 
            1.0 / (1.0 + distance) : 
            (score || 1.0);
          
          memories.push({
            id: id,
            content: content,
            similarity: similarity,
            metadata,
          });
        }
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

      const escapedId = args.id.replace(/"/g, '\\"');
      await this.client.delete({
        collection_name: collectionName,
        expr: `id == "${escapedId}"`,
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
      // Check if Milvus is available
      const status = await this.client.checkHealth();
      if (!status.isHealthy) {
        throw new Error(`Milvus server at ${this.host}:${this.port} is not healthy`);
      }
      console.log(`Connected to Milvus server at ${this.host}:${this.port}`);
      
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
    } catch (error) {
      console.error(`Failed to connect to Milvus: ${error.message}`);
      process.exit(1);
    }
  }
}

async function main() {
  try {
    const server = new MilvusMCPServer(argv.host, argv.port, argv.collection, argv['embedding-model']);
    await server.run();
  } catch (error) {
    console.error('Failed to start MCP server:', error);
    process.exit(1);
  }
}

main().catch(error => {
  console.error('Unhandled error:', error);
  process.exit(1);
}); 