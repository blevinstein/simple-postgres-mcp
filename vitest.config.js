import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    testTimeout: 30000, // 30 seconds for integration tests that involve network calls
    hookTimeout: 10000, // 10 seconds for setup/teardown hooks
    setupFiles: [],
    env: {
      NODE_ENV: 'test',
    },
    globals: false,
    environment: 'node',
  },
  esbuild: {
    target: 'node18'
  }
}); 