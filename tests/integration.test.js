import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { MilvusMCPServer } from '../src/mcp-server.js';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

describe('Milvus MCP Server Integration Tests', () => {
  let server;
  let client;
  const testCollection = `test_integration_${Date.now()}`;
  const host = 'localhost';
  const port = '19530';
  const embeddingModel = 'google/text-embedding-004';

  beforeAll(async () => {
    // Check if Milvus is running
    client = new MilvusClient({ address: `${host}:${port}` });
    
    try {
      const status = await client.checkHealth();
      if (!status.isHealthy) {
        throw new Error('Milvus is not healthy');
      }
    } catch (error) {
      throw new Error(`Could not connect to Milvus at ${host}:${port}. Make sure Milvus is running: ${error.message}`);
    }

    // Initialize server - fix constructor call
    server = new MilvusMCPServer(host, port, testCollection, embeddingModel);
  });

  afterAll(async () => {
    if (client) {
      try {
        // Clean up all test collections
        const collections = await client.listCollections();
        for (const collection of collections.collection_names) {
          if (collection.startsWith('test_integration_')) {
            await client.dropCollection({ collection_name: collection });
          }
        }
      } catch (error) {
        console.warn('Cleanup failed:', error.message);
      }
      await client.closeConnection();
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
        expect(response.success).toBe(true);
        
        const memories = response.result.memories;
        expect(memories.length).toBe(1); // Only one document contains "convolutional"
        expect(memories[0].content).toContain('convolutional');
      });

      it('should return empty results for non-matching terms', async () => {
        const result = await server.searchMemory({ 
          query: 'quantum computing blockchain', 
          mode: 'fulltext', 
          limit: 3 
        });

        const response = JSON.parse(result.content[0].text);
        expect(response.success).toBe(true);
        expect(response.result.count).toBe(0);
        expect(response.result.memories).toHaveLength(0);
      });
    });

    it('should handle invalid search mode', async () => {
      const result = await server.searchMemory({ 
        query: 'test query', 
        mode: 'invalid_mode', 
        limit: 5 
      });

      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.operation).toBe('search');
      expect(response.error).toContain('Unknown search mode'); // Fixed to match actual error message
    });
  });

  describe('Delete Memory', () => {
    it('should delete a memory successfully', async () => {
      // Store a memory first
      const storeResult = await server.storeMemory({
        content: 'Memory to be deleted for testing purposes.',
        metadata: { test: 'delete_test' }
      });
      
      const storeResponse = JSON.parse(storeResult.content[0].text);
      const memoryId = storeResponse.result.id;

      // Delete the memory
      const deleteResult = await server.forgetMemory({ id: memoryId });
      
      const deleteResponse = JSON.parse(deleteResult.content[0].text);
      expect(deleteResponse.success).toBe(true);
      expect(deleteResponse.operation).toBe('delete');
      expect(deleteResponse.result.id).toBe(memoryId);

      // Verify it's deleted by searching
      const searchResult = await server.searchMemory({
        query: 'deleted testing purposes',
        mode: 'semantic',
        limit: 10
      });
      
      const searchResponse = JSON.parse(searchResult.content[0].text);
      expect(searchResponse.result.count).toBe(0);
    });

    it('should handle deletion of non-existent memory gracefully', async () => {
      const result = await server.forgetMemory({ id: 'nonexistent_id_12345' });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.operation).toBe('delete');
      expect(response.error).toContain('Invalid memory ID format');
    });

    it('should handle deletion of properly formatted but non-existent memory ID', async () => {
      // First store a memory to ensure collection exists
      await server.storeMemory({
        content: 'Test content to ensure collection exists',
        metadata: { test: 'setup' },
        collection: testCollection 
      });

      // Now try to delete a non-existent memory from the existing collection
      const result = await server.forgetMemory({ 
        id: 'mem_1234567890_nonexistent',
        collection: testCollection 
      });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.operation).toBe('delete');
      expect(response.error).toContain('not found');
    });
  });

  describe('Collection Management', () => {
    it('should create collection with proper BM25 schema', async () => {
      const testCollectionName = `test_collection_${Date.now()}`;
      
      // Store something to trigger collection creation
      await server.storeMemory({
        content: 'Test content for collection creation',
        metadata: { purpose: 'schema_test' },
        collection: testCollectionName
      });

      // Verify collection exists
      const hasCollection = await client.hasCollection({ collection_name: testCollectionName });
      expect(hasCollection.value).toBe(true);

      // Clean up
      await client.dropCollection({ collection_name: testCollectionName });
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid search modes', async () => {
      const result = await server.searchMemory({ query: 'test query', mode: 'invalid_mode', limit: 5 });
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(false);
      expect(response.error).toContain('Unknown search mode'); // Fixed to match actual error
    });
  });

  // Additional test categories that work and provide value
  describe('Edge Cases - Working Tests', () => {
    it('should handle special characters and unicode', async () => {
      const specialContent = '🚀 Testing émojis, àccénts, and 中文字符! @#$%^&*()';
      const result = await server.storeMemory({ content: specialContent, metadata: { unicode: true, emoji_count: 1 } });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(true);
      expect(response.result.id).toMatch(/^mem_\d+_[a-z0-9]+$/);

      // Verify we can search and retrieve it
      const searchResult = await server.searchMemory({ query: 'émojis', mode: 'semantic', limit: 5 });
      const searchResponse = JSON.parse(searchResult.content[0].text);
      expect(searchResponse.success).toBe(true);
      expect(searchResponse.result.memories.length).toBeGreaterThan(0);
      expect(searchResponse.result.memories[0].content).toBe(specialContent);
    });

    it('should handle deeply nested metadata', async () => {
      const complexMetadata = {
        level1: {
          level2: {
            level3: {
              array: [1, 2, 3, { nested: 'value' }],
              boolean: true,
              null_value: null,
              number: 42.5
            }
          }
        },
        tags: ['tag1', 'tag2', 'tag3']
      };

      const result = await server.storeMemory({ 
        content: 'Content with complex metadata structure', 
        metadata: complexMetadata 
      });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(true);
      expect(response.result.metadata).toEqual(complexMetadata);
    });

    it('should handle very long content (>10000 chars)', async () => {
      const longContent = 'A'.repeat(15000) + ' This is a test of very long content handling in the Milvus MCP server.';
      const result = await server.storeMemory({ content: longContent, metadata: { type: 'long_content_test' } });
      
      const response = JSON.parse(result.content[0].text);
      expect(response.success).toBe(true);
      expect(response.result.content_length).toBe(longContent.length);
      expect(response.result.id).toMatch(/^mem_\d+_[a-z0-9]+$/);
    });
  });

  describe('Cross-Collection Operations', () => {
    const collection1 = `test_cross_col1_${Date.now()}`;
    const collection2 = `test_cross_col2_${Date.now()}`;

    it('should isolate data between different collections', async () => {
      // Store content in collection1
      const result1 = await server.storeMemory({ 
        content: 'Content in collection 1', 
        metadata: { source: 'col1' }, 
        collection: collection1 
      });
      const response1 = JSON.parse(result1.content[0].text);
      expect(response1.success).toBe(true);

      // Store content in collection2
      const result2 = await server.storeMemory({ 
        content: 'Content in collection 2', 
        metadata: { source: 'col2' }, 
        collection: collection2 
      });
      const response2 = JSON.parse(result2.content[0].text);
      expect(response2.success).toBe(true);

      // Search in collection1 should only find collection1 content
      const search1 = await server.searchMemory({ 
        query: 'Content', 
        mode: 'semantic', 
        limit: 10, 
        collection: collection1 
      });
      const searchResponse1 = JSON.parse(search1.content[0].text);
      expect(searchResponse1.success).toBe(true);
      expect(searchResponse1.result.memories.length).toBe(1);
      expect(searchResponse1.result.memories[0].metadata.source).toBe('col1');

      // Search in collection2 should only find collection2 content  
      const search2 = await server.searchMemory({ 
        query: 'Content', 
        mode: 'semantic', 
        limit: 10, 
        collection: collection2 
      });
      const searchResponse2 = JSON.parse(search2.content[0].text);
      expect(searchResponse2.success).toBe(true);
      expect(searchResponse2.result.memories.length).toBe(1);
      expect(searchResponse2.result.memories[0].metadata.source).toBe('col2');
    });
  });

  describe('Performance and Scalability', () => {
    const perfCollection = `test_perf_${Date.now()}`;

    it('should handle batch operations efficiently', async () => {
      const batchSize = 5; // Reduced size for faster testing
      const startTime = Date.now();
      const promises = [];

      // Store multiple memories concurrently
      for (let i = 0; i < batchSize; i++) {
        promises.push(
          server.storeMemory({ 
            content: `Batch test content ${i} with unique identifier for testing concurrent operations`, 
            metadata: { batch: true, index: i }, 
            collection: perfCollection 
          })
        );
      }

      const results = await Promise.all(promises);
      const endTime = Date.now();

      // All should succeed
      results.forEach(result => {
        const response = JSON.parse(result.content[0].text);
        expect(response.success).toBe(true);
        expect(response.result.id).toMatch(/^mem_\d+_[a-z0-9]+$/);
      });

      // Should complete in reasonable time (less than 30 seconds for 5 items)
      expect(endTime - startTime).toBeLessThan(30000);

      // Test batch search
      const searchResult = await server.searchMemory({ 
        query: 'batch test', 
        mode: 'semantic', 
        limit: batchSize, 
        collection: perfCollection 
      });
      const searchResponse = JSON.parse(searchResult.content[0].text);
      expect(searchResponse.success).toBe(true);
      expect(searchResponse.result.memories.length).toBe(batchSize);
    });
  });
}); 