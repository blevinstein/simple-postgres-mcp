#!/usr/bin/env node

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { PostgresMCPServer } from './mcp-server.js';

const argv = yargs(hideBin(process.argv))
  .option('host', {
    type: 'string',
    default: 'localhost',
    description: 'PostgreSQL server host'
  })
  .option('port', {
    type: 'string',
    default: '5432',
    description: 'PostgreSQL server port'
  })
  .option('database', {
    type: 'string',
    default: 'mcp_memories',
    description: 'PostgreSQL database name'
  })
  .option('collection', {
    type: 'string',
    description: 'Default table/collection name (if not provided, must be specified per tool call)'
  })
  .option('embedding-model', {
    type: 'string',
    default: 'google/text-embedding-004',
    description: 'Default embedding model (e.g., openai/text-embedding-3-small, vertex/text-embedding-005, google/text-embedding-004)'
  })
  .help()
  .argv;

async function main() {
  try {
    const server = new PostgresMCPServer(
      argv.host,
      argv.port,
      argv.database,
      argv.collection,
      argv['embedding-model']
    );
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