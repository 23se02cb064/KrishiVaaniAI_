const { createProxyMiddleware } = require('http-proxy-middleware');

const API_URL = process.env.RENDER_BACKEND_URL;

if (!API_URL) {
  throw new Error('RENDER_BACKEND_URL environment variable is not configured');
}

// The same origin allow-list the Flask backend uses, so a proxy-generated
// error page carries the same CORS contract as a backend-generated one.
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map((o) => o.trim())
  .filter(Boolean);

function applyCors(req, res) {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  } else if (!res.getHeader('Access-Control-Allow-Origin')) {
    // Same-origin browser requests (the normal case through this proxy) have
    // no Origin header, so there is nothing to allow.
    res.setHeader('Access-Control-Allow-Origin', '*');
  }
  res.setHeader('Vary', 'Origin');
}

const apiProxy = createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
  timeout: 30000,
  proxyTimeout: 30000,
  pathRewrite: (path, req) => {
    const originalPath = req.query?.path;

    if (typeof originalPath === 'string') {
      return `/api/${originalPath}`;
    }

    return path;
  }
});

module.exports = (req, res) => {
  // Apply CORS headers BEFORE the proxy runs. http-proxy-middleware writes its
  // own error body on failure, which sets res.headersSent before this module's
  // error callback is reached; checking headersSent afterwards therefore
  // misses it. Setting the headers up front guarantees the 502/504 error page
  // carries the same CORS contract as a backend-generated response, so the
  // browser does not block it as a CORS violation.
  applyCors(req, res);
  apiProxy(req, res, (err) => {
    console.error('Proxy error:', err);

    if (!res.headersSent) {
      res.status(502).json({ error: 'Backend proxy error' });
    }
  });
};