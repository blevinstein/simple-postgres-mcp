#!/usr/bin/env node

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { CallToolRequestSchema, ListToolsRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const yargs = require('yargs');
const { hideBin } = require('yargs/helpers');
const { MilvusClient } = require('@zilliz/milvus2-sdk-node');

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
  constructor(host, port, fixedCollection, defaultEmbeddingModel) {
    this.host = host;
    this.port = port;
    this.fixedCollection = fixedCollection;
    this.defaultEmbeddingModel = defaultEmbeddingModel;
    this.embedText = null;
    
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
      const polytokenizer = await import('polytokenizer');
      this.embedText = polytokenizer.embedText;
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
                embedding_model: {
                  type: 'string',
                  description: `Embedding model to use (optional, default: ${this.defaultEmbeddingModel}). Examples: openai/text-embedding-3-small, vertex/text-embedding-005, google/text-embedding-004`
                }
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
                embedding_model: {
                  type: 'string',
                  description: `Embedding model to use for semantic search (optional, default: ${this.defaultEmbeddingModel}). Examples: openai/text-embedding-3-small, vertex/text-embedding-005, google/text-embedding-004`
                }
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
    return args.collection || this.defaultCollection;
  }

  getEmbeddingModel(args) {
    return args.embedding_model || this.defaultEmbeddingModel;
  }

  async ensureCollection(collectionName) {
    if (!collectionName) {
      throw new Error('Collection name is required either as argument or default');
    }

    const hasCollection = await this.client.hasCollection({
      collection_name: collectionName,
    });

    if (!hasCollection.value) {
      throw new Error(`Collection '${collectionName}' does not exist`);
    }

    await this.client.loadCollection({
      collection_name: collectionName,
    });
  }

  async generateEmbedding(text, embeddingModel) {
    try {
      const embedText = await this.loadEmbedText();
      const result = await embedText(embeddingModel, text);
      return result.vector;
    } catch (error) {
      throw new Error(`Failed to generate embedding with ${embeddingModel}: ${error.message}`);
    }
  }

  async storeMemory(args) {
    try {
      const collectionName = this.getCollectionName(args);
      const embeddingModel = this.getEmbeddingModel(args);
      
      await this.ensureCollection(collectionName);

      const embedding = await this.generateEmbedding(args.content, embeddingModel);
      
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

      if (mode === 'semantic') {
        const embeddingModel = this.getEmbeddingModel(args);
        const queryEmbedding = await this.generateEmbedding(args.query, embeddingModel);
        
        results = await this.client.search({
          collection_name: collectionName,
          vectors: [queryEmbedding],
          search_params: {
            anns_field: 'embedding',
            topk: limit,
            metric_type: 'L2',
            params: JSON.stringify({ nprobe: 10 }),
          },
          output_fields: ['id', 'content'],
        });
      } else if (mode === 'fulltext') {
        results = await this.client.query({
          collection_name: collectionName,
          expr: `content like "%${args.query}%"`,
          output_fields: ['id', 'content'],
          limit: limit,
        });
      } else {
        throw new Error(`Unknown search mode: ${mode}`);
      }

      const memories = [];
      if (results.results && results.results.length > 0) {
        for (const hit of results.results) {
          const similarity = hit.distance !== undefined ? 
            1.0 / (1.0 + hit.distance) : 
            (hit.score || 1.0);
          
          memories.push({
            id: hit.id,
            content: hit.content,
            similarity: similarity
          });
        }
      } else if (results.data) {
        for (const item of results.data) {
          memories.push({
            id: item.id,
            content: item.content,
            similarity: 1.0
          });
        }
      }

      const embeddingInfo = mode === 'semantic' ? ` using embedding model: ${this.getEmbeddingModel(args)}` : '';
      return {
        content: [
          {
            type: 'text',
            text: `Found ${memories.length} memories using ${mode} search${embeddingInfo}:\n\n` +
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
        expr: `id == "${args.id}"`,
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
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
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