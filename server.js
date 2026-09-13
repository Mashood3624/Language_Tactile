const http = require('http');
const fs = require('fs');
const path = require('path');

const root = process.cwd();
const types = {
  '.html': 'text/html; charset=utf-8',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg'
};

http.createServer((request, response) => {
  let url = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  if (url === '/') url = '/index.html';

  const file = path.resolve(root, `.${url}`);
  if (!file.startsWith(`${root}${path.sep}`)) {
    response.writeHead(403);
    response.end();
    return;
  }

  fs.readFile(file, (error, data) => {
    if (error) {
      response.writeHead(404);
      response.end('Not found');
      return;
    }

    response.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    response.end(data);
  });
}).listen(4173, '127.0.0.1');
