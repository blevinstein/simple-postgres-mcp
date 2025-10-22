-- Initialize PostgreSQL database with pgvector extension
-- This script runs automatically when the container is first created

CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
