import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const here = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = resolve(here, '../..');
const dashboardDataRoot = resolve(repoRoot, 'dashboard_data');

function contentType(pathname: string): string {
  if (extname(pathname) === '.json') return 'application/json; charset=utf-8';
  return 'application/octet-stream';
}

export default defineConfig({
  root: '.',
  base: './',
  plugins: [
    {
      name: 'dashboard-data-v2-dev-server',
      configureServer(server) {
        server.middlewares.use('/dashboard_data', (request, response, next) => {
          const requestPath = decodeURIComponent(request.url?.split('?')[0] ?? '/');
          const resolved = resolve(dashboardDataRoot, `.${normalize(requestPath)}`);
          if (!resolved.startsWith(dashboardDataRoot) || !existsSync(resolved) || !statSync(resolved).isFile()) {
            next();
            return;
          }
          response.statusCode = 200;
          response.setHeader('Content-Type', contentType(resolved));
          createReadStream(resolved).pipe(response);
        });
      },
    },
  ],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  test: {
    environment: 'jsdom',
  },
});
