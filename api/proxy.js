const { createProxyMiddleware } = require('http-proxy-middleware');

const API_URL = process.env.RENDER_BACKEND_URL;

if (!API_URL) {
  throw new Error('RENDER_BACKEND_URL environment variable is not configured');
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
  apiProxy(req, res, (err) => {
    console.error('Proxy error:', err);

    if (!res.headersSent) {
      res.status(502).json({ error: 'Backend proxy error' });
    }
  });
};