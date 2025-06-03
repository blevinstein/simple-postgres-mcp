import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';

describe('Milvus MCP Server Integration Tests', () => {
  let server;
  let client;
  let MilvusMCPServer, MilvusClient;
  const testCollection = `test_integration_${Date.now()}`;
  const host = 'localhost';
  const port = '19530';
  const embeddingModel = 'google/text-embedding-004';

  beforeAll(async () => {
    // Load CommonJS modules dynamically
    const serverModule = await import('../src/milvusMCPServer.js');
    const clientModule = await import('@zilliz/milvus2-sdk-node');
    MilvusMCPServer = serverModule.MilvusMCPServer;
    MilvusClient = clientModule.MilvusClient;

    // Check if Milvus is running
    client = new MilvusClient({ address: `${host}:${port}` });
    
    try {
      const status = await client.checkHealth();
      if (!status.isHealthy) {
        throw new Error(`Milvus server at ${host}:${port} is not healthy`);
      }
    } catch (error) {
      throw new Error(`Cannot connect to Milvus at ${host}:${port}. Please ensure Milvus is running. Error: ${error.message}`);
    }

    // Initialize MCP server
    server = new MilvusMCPServer(host, port, testCollection, embeddingModel);
  });

  afterAll(async () => {
    // Clean up test collection
    try {
      const hasCollection = await client.hasCollection({ collection_name: testCollection });
      if (hasCollection.value) {
        await client.dropCollection({ collection_name: testCollection });
      }
    } catch (error) {
      console.warn('Failed to clean up test collection:', error.message);
    }
  });

  beforeEach(async () => {
    // Ensure collection exists and is clean for each test
    try {
      const hasCollection = await client.hasCollection({ collection_name: testCollection });
      if (hasCollection.value) {
        await client.dropCollection({ collection_name: testCollection });
      }
    } catch (error) {
      // Collection might not exist, which is fine
    }
  });

  describe('Store Memory', () => {
    it('should store a memory with metadata successfully', async () => {
      const content = 'Machine learning algorithms require large datasets to train effectively.';
      const metadata = { topic: 'ML', type: 'educational' };

      const result = await server.storeMemory({ content, metadata });
      
      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe('text');
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(true);
      expect(response.operation).toBe('store');
      expect(response.result.id).toMatch(/^mem_\d+_[a-z0-9]+$/);
      expect(response.result.collection).toBe(testCollection);
      expect(response.result.content_length).toBe(content.length);
      expect(response.result.embedding_dimensions).toBe(768);
      expect(response.result.embedding_model).toBe(embeddingModel);
      expect(response.result.metadata).toEqual(metadata);
      expect(response.result.created_at).toBeDefined();
    });

    it('should store a memory without metadata', async () => {
      const content = 'Neural networks are inspired by biological neural systems.';

      const result = await server.storeMemory({ content });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(true);
      expect(response.result.metadata).toEqual({});
    });

    it('should handle invalid embedding model gracefully', async () => {
      const invalidServer = new MilvusMCPServer(host, port, testCollection, 'invalid/model');
      const content = 'Test content';

      const result = await invalidServer.storeMemory({ content });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.operation).toBe('store');
      expect(response.error).toContain('not found in EMBEDDING_DIMENSIONS');
    });
  });

  describe('Search Memory', () => {
    beforeEach(async () => {
      // Add test data for search tests
      const testData = [
        {
          content: 'Machine learning algorithms require large datasets to train effectively. Neural networks are particularly data-hungry models.',
          metadata: { topic: 'ML Training', type: 'educational', complexity: 'intermediate' }
        },
        {
          content: 'Computer vision applications use convolutional neural networks to process and analyze visual data from images and videos.',
          metadata: { topic: 'Computer Vision', type: 'technical', field: 'AI research' }
        },
        {
          content: 'Natural language processing enables computers to understand, interpret, and generate human language through various algorithms.',
          metadata: { topic: 'NLP', type: 'research', field: 'linguistics' }
        }
      ];

      for (const data of testData) {
        await server.storeMemory(data);
      }
      
      // Wait a moment for data to be available
      await new Promise(resolve => setTimeout(resolve, 100));
    });

    describe('Semantic Search', () => {
      it('should find relevant memories using semantic search', async () => {
        const result = await server.searchMemory({ 
          query: 'neural networks machine learning', 
          mode: 'semantic', 
          limit: 3 
        });

        const response = JSON.parse(result.content[0].text);
        expect(response.success).toBe(true);
        expect(response.operation).toBe('search');
        expect(response.result.query).toBe('neural networks machine learning');
        expect(response.result.mode).toBe('semantic');
        expect(response.result.count).toBeGreaterThan(0);
        expect(response.result.memories).toBeInstanceOf(Array);
        
        const memories = response.result.memories;
        expect(memories.length).toBeGreaterThan(0);
        
        // Check memory structure
        const memory = memories[0];
        expect(memory.id).toMatch(/^mem_\d+_[a-z0-9]+$/);
        expect(memory.content).toBeDefined();
        expect(memory.similarity).toBeTypeOf('number');
        expect(memory.metadata).toBeTypeOf('object');
        expect(memory.created_at).toBeDefined();
      });
    });

    describe('Fulltext Search', () => {
      it('should find relevant memories using BM25 fulltext search', async () => {
        const result = await server.searchMemory({ 
          query: 'neural networks', 
          mode: 'fulltext', 
          limit: 3 
        });

        const response = JSON.parse(result.content[0].text);
        expect(response.success).toBe(true);
        expect(response.operation).toBe('search');
        expect(response.result.query).toBe('neural networks');
        expect(response.result.mode).toBe('fulltext');
        expect(response.result.count).toBeGreaterThan(0);
        
        const memories = response.result.memories;
        expect(memories.length).toBeGreaterThan(0);
        
        // All returned memories should contain the search terms
        memories.forEach(memory => {
          expect(memory.content.toLowerCase()).toMatch(/neural|networks/);
        });
      });

      it('should handle specific term searches', async () => {
        const result = await server.searchMemory({ 
          query: 'convolutional', 
          mode: 'fulltext', 
          limit: 3 
        });

        const response = JSON.parse(result.content[0].text);
        const memories = response.result.memories;
        
        expect(memories.length).toBe(1);
        expect(memories[0].content).toContain('convolutional');
      });

      it('should return empty results for non-matching terms', async () => {
        const result = await server.searchMemory({ 
          query: 'nonexistent term xyz', 
          mode: 'fulltext', 
          limit: 3 
        });

        const response = JSON.parse(result.content[0].text);
        expect(response.result.count).toBe(0);
        expect(response.result.memories).toEqual([]);
      });
    });

    it('should handle invalid search mode', async () => {
      const result = await server.searchMemory({ 
        query: 'test query', 
        mode: 'invalid_mode', 
        limit: 3 
      });

      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.operation).toBe('search');
      expect(response.error).toContain('Unknown search mode');
    });
  });

  describe('Delete Memory', () => {
    it('should delete a memory successfully', async () => {
      // First store a memory
      const storeResult = await server.storeMemory({ 
        content: 'This memory will be deleted', 
        metadata: { test: true } 
      });
      
      const storeResponse = JSON.parse(storeResult.content[0].text);
      const memoryId = storeResponse.result.id;

      // Delete the memory
      const deleteResult = await server.forgetMemory({ id: memoryId });
      
      const deleteResponse = JSON.parse(deleteResult.content[0].text);
      expect(deleteResponse.success).toBe(true);
      expect(deleteResponse.operation).toBe('delete');
      expect(deleteResponse.result.id).toBe(memoryId);
      expect(deleteResponse.result.collection).toBe(testCollection);

      // Verify memory is deleted by searching
      const searchResult = await server.searchMemory({ 
        query: 'This memory will be deleted', 
        mode: 'semantic', 
        limit: 10 
      });
      
      const searchResponse = JSON.parse(searchResult.content[0].text);
      const foundMemory = searchResponse.result.memories.find(m => m.id === memoryId);
      expect(foundMemory).toBeUndefined();
    });

    it('should handle deletion of non-existent memory gracefully', async () => {
      const result = await server.forgetMemory({ id: 'nonexistent_id_12345' });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(true); // Milvus doesn't error on non-existent deletions
      expect(response.operation).toBe('delete');
      expect(response.result.id).toBe('nonexistent_id_12345');
    });
  });

  describe('Collection Management', () => {
    it('should create collection with proper BM25 schema', async () => {
      // Trigger collection creation by storing data
      await server.storeMemory({ 
        content: 'Test content for schema validation',
        metadata: { test: true }
      });

      // Verify collection exists and has correct schema
      const hasCollection = await client.hasCollection({ collection_name: testCollection });
      expect(hasCollection.value).toBe(true);

      const description = await client.describeCollection({ collection_name: testCollection });
      const schema = description.schema;
      
      // Check required fields exist
      const fieldNames = schema.fields.map(f => f.name);
      expect(fieldNames).toContain('id');
      expect(fieldNames).toContain('content');
      expect(fieldNames).toContain('embedding');
      expect(fieldNames).toContain('sparse');

      // Check BM25 function exists
      expect(schema.functions).toHaveLength(1);
      expect(schema.functions[0].type).toBe('BM25');
      expect(schema.functions[0].name).toBe('content_bm25_emb');
    });
  });

  describe('Error Handling', () => {
    it('should handle missing API key gracefully', async () => {
      // This test assumes no GEMINI_API_KEY is set or an invalid one is used
      const originalKey = process.env.GEMINI_API_KEY;
      delete process.env.GEMINI_API_KEY;

      try {
        const serverWithoutKey = new MilvusMCPServer(host, port, testCollection + '_no_key', embeddingModel);
        const result = await serverWithoutKey.storeMemory({ content: 'Test content' });
        
        const response = JSON.parse(result.content[0].text);
        if (!response.success) {
          expect(response.error).toMatch(/API key not configured|API key not found/);
        }
      } finally {
        // Restore original key
        if (originalKey) {
          process.env.GEMINI_API_KEY = originalKey;
        }
      }
    });

    it('should handle invalid collection names', async () => {
      const invalidServer = new MilvusMCPServer(host, port, '', embeddingModel);
      const result = await invalidServer.storeMemory({ content: 'Test content' });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.error).toContain('Collection name is required');
    });
  });
}); 