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
      
      if (!EMBEDDING_DIMENSIONS) {
        throw new Error('EMBEDDING_DIMENSIONS not available from polytokenizer');
      }
      
      const availableModels = Object.keys(EMBEDDING_DIMENSIONS);
      const expectedDim = EMBEDDING_DIMENSIONS[this.embeddingModel];
      
      if (!expectedDim) {
        throw new Error(`Model '${this.embeddingModel}' not found in EMBEDDING_DIMENSIONS. Available models: ${availableModels.join(', ')}`);
      }

      try {
        // Validate field definitions before creating collection
        if (!expectedDim || typeof expectedDim !== 'number' || expectedDim <= 0) {
          throw new Error(`Invalid embedding dimension: ${expectedDim} (type: ${typeof expectedDim})`);
        }
        
        // Use explicit schema format
        const fields = [
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
        ];
        
        // Validate fields before passing to SDK
        for (const field of fields) {
          if (field.data_type === 'FloatVector' && (!field.dim || typeof field.dim !== 'number')) {
            throw new Error(`FloatVector field '${field.name}' missing or invalid dim property: ${field.dim}`);
          }
          if (field.data_type === 'VarChar' && (!field.max_length || typeof field.max_length !== 'number')) {
            throw new Error(`VarChar field '${field.name}' missing or invalid max_length property: ${field.max_length}`);
          }
        }
        
        await this.client.createCollection({
          collection_name: collectionName,
          fields: fields,
          enable_dynamic_field: true,
        });
        
        // Create index for the embedding field
        try {
          await this.client.createIndex({
            collection_name: collectionName,
            field_name: 'embedding',
            index_type: 'FLAT',
            metric_type: 'L2',
          });
        } catch (indexError) {
          throw new Error(`Failed to create index for collection ${collectionName}: ${indexError.message}`);
        }
      } catch (createError) {
        throw new Error(`Failed to create collection ${collectionName}: ${createError.message}`);
      }
    }

    try {
      await this.client.loadCollection({
        collection_name: collectionName,
      });
    } catch (loadError) {
      throw new Error(`Failed to load collection ${collectionName}: ${loadError.message}`);
    }
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
      
      try {
        await this.ensureCollection(collectionName);
      } catch (ensureError) {
        throw new Error(`Collection setup failed: ${ensureError.message}`);
      }

      // Load embedding dimensions and validate
      const { EMBEDDING_DIMENSIONS } = await this.loadPolytokenizer();
      
      if (!EMBEDDING_DIMENSIONS) {
        throw new Error('EMBEDDING_DIMENSIONS not available from polytokenizer');
      }
      
      const availableModels = Object.keys(EMBEDDING_DIMENSIONS);
      const expectedDim = EMBEDDING_DIMENSIONS[this.embeddingModel];
      
      if (!expectedDim) {
        throw new Error(`Model '${this.embeddingModel}' not found in EMBEDDING_DIMENSIONS. Available models: ${availableModels.join(', ')}`);
      }
      
      const embedding = await this.generateEmbedding(args.content);
      
      // Check if embedding dimensions match what we expect
      if (embedding.length !== expectedDim) {
        throw new Error(`Embedding dimension mismatch: got ${embedding.length}, expected ${expectedDim} for model ${this.embeddingModel}`);
      }
      
      // Validate embedding for NaN values
      const hasNaN = embedding.some(val => isNaN(val) || !isFinite(val));
      if (hasNaN) {
        throw new Error(`Generated embedding contains NaN or infinite values. Embedding length: ${embedding.length}`);
      }
      
      // Additional validation - ensure all values are proper numbers
      const normalizedEmbedding = embedding.map(val => {
        if (typeof val !== 'number' || isNaN(val) || !isFinite(val)) {
          throw new Error(`Invalid embedding value: ${val} (type: ${typeof val})`);
        }
        return Number(val);
      });
      
      // Additional validation - log first few values for debugging
      const firstFewValues = normalizedEmbedding.slice(0, 5);
      const lastFewValues = normalizedEmbedding.slice(-5);
      
      // Generate a simple ID
      const generatedId = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Use dynamic fields since enableDynamicField is true
      const coreData = [{
        id: generatedId,
        embedding: normalizedEmbedding,
        content: args.content,
        metadata_json: JSON.stringify(args.metadata || {}),
        created_at: new Date().toISOString(),
      }];

      let result;
      try {
        result = await this.client.insert({
          collection_name: collectionName,
          data: coreData,
        });
      } catch (insertError) {
        throw new Error(`Insert operation failed: ${insertError.message}`);
      }

      await this.client.flush({
        collection_names: [collectionName],
      });

      // Verify the data was inserted by querying it back
      let verificationResult;
      try {
        verificationResult = await this.client.query({
          collection_name: collectionName,
          expr: `id == "${generatedId}"`,
          output_fields: ['id', 'content', 'created_at'],
          limit: 1,
        });
      } catch (verifyError) {
        verificationResult = { error: verifyError.message };
      }

      return {
        content: [
          {
            type: 'text',
            text: `Successfully stored memory in collection: ${collectionName}
- Generated ID: ${generatedId}
- Content length: ${args.content.length} characters
- Embedding dimensions: ${normalizedEmbedding.length} (expected: ${expectedDim})
- Embedding model: ${this.embeddingModel}
- Metadata fields: ${Object.keys(args.metadata || {}).join(', ') || 'none'}
- Embedding first 3: [${firstFewValues.slice(0,3).join(', ')}]
- Embedding type: ${typeof normalizedEmbedding[0]}
- Insert result: ${JSON.stringify(result, null, 2)}
- Verification query: ${verificationResult.error ? 'FAILED - ' + verificationResult.error : 'SUCCESS - Found ' + (verificationResult.data?.length || 0) + ' records'}`
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

      // Ensure any pending data is flushed
      await this.client.flush({
        collection_names: [collectionName],
      });

      const mode = args.mode || 'semantic';
      const limit = args.limit || 10;
      let results;

      // Get collection statistics for debugging
      const stats = await this.client.getCollectionStatistics({
        collection_name: collectionName,
      });

      // Get collection fields to extract all metadata
      const collectionInfo = await this.client.describeCollection({
        collection_name: collectionName,
      });
      
      const allFields = collectionInfo.schema.fields.map(field => field.name);
      const outputFields = ['id', 'content', 'metadata_json', 'created_at'];

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
          const { id, content, metadata_json, created_at, embedding, distance, score } = hit;
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
      } else if (results.data) {
        for (const item of results.data) {
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

      // First check if the memory exists
      const checkResults = await this.client.query({
        collection_name: collectionName,
        expr: `id == "${args.id}"`,
        output_fields: ['id', 'created_at'],
        limit: 1,
      });

      if (!checkResults.data || checkResults.data.length === 0) {
        return {
          content: [
            {
              type: 'text',
              text: `Memory with ID '${args.id}' not found in collection: ${collectionName}`
            }
          ],
          isError: true
        };
      }

      // Delete using the auto-generated ID
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
      // Check if Milvus is available
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