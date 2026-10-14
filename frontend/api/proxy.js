// Vercel serverless proxy: forwards /api/* to the Render backend.
//
// Deliberately dependency-free. http-proxy-middleware ships an ESM-only build
// that Vercel's Node runtime cannot require() (ERR_REQUIRE_ESM), so this
// handler uses Node's built-in fetch() instead. That removes the
// module-format conflict and the 13-package dependency tree entirely.
//
// It is also ESM (export default), not CommonJS (module.exports), because
// Vercel's @vercel/node builder copies the ROOT package.json into every
// function directory, and this repo's root package.json carries
// "type": "module" (it is a Vite/ESM app). Under "type": "module" a CommonJS
// module.exports is silently ignored, so a CommonJS proxy would export
// nothing and every /api/* request would 500.
//
// Behaviour preserved from the previous implementation:
//   - every HTTP method, query string, and request body (including multipart
//     uploads) is forwarded verbatim to RENDER_BACKEND_URL
//   - the backend's response status, headers, and body are returned unchanged
//   - CORS headers are applied before the call, so the proxy's own error page
//     is not blocked by the browser as a CORS violation
//   - proxy failures return JSON, never a crash
//   - secrets and request bodies are never logged

const API_URL = process.env.RENDER_BACKEND_URL;

if (!API_URL) {
  throw new Error('RENDER_BACKEND_URL environment variable is not configured');
}

// Strip the trailing slash so path concatenation is predictable.
const BASE = API_URL.replace(/\/+$/, '');

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
  // Preflight requests need a short allow-list to succeed.
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods',
      'GET, POST, PUT, PATCH, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers',
      'Content-Type, Authorization, X-Requested-With');
    res.setHeader('Access-Control-Max-Age', '86400');
    res.writeHead(204);
    res.end();
    return true;
  }
  return false;
}

function buildTarget(req) {
  // Forward the incoming path and query string verbatim. The incoming path on
  // Vercel is already /api/... because vercel.json routes /api/(.*) here.
  const incoming = req.url || '/';
  return BASE + incoming;
}

function forwardHeaders(req) {
  const headers = {};
  for (const [key, value] of Object.entries(req.headers || {})) {
    // hop-by-hop headers must not be forwarded.
    if (key === 'host' || key === 'content-length') continue;
    headers[key] = value;
  }
  return headers;
}

async function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

// ESM export. Vercel's @vercel/node builder copies the ROOT package.json
// (which carries "type": "module" because this is a Vite/ESM app) into every
// function directory. Under "type": "module" a CommonJS module.exports is
// silently ignored, so the deployed function would export nothing and every
// /api/* request would 500. export default is what the Vercel Node runtime
// loader calls, so it works regardless of which package.json is present.
export default async (req, res) => {
  // CORS must be applied before any downstream work, because the proxy's own
  // error response is the one the browser will see if the backend is
  // unreachable. http-proxy-middleware used to write its own body first and
  // bypass this, which is exactly the "missing Access-Control-Allow-Origin"
  // error reported in production.
  if (applyCors(req, res)) return;

  const target = buildTarget(req);
  let body = undefined;
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    body = await readBody(req);
  }

  let upstream;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers: forwardHeaders(req),
      body: body && body.length ? body : undefined,
      // The backend can be slow on the free tier; give it room but not forever.
      signal: AbortSignal.timeout(30000),
    });
  } catch (err) {
    // Connection refused, DNS failure, or timeout. Return JSON with CORS so
    // the browser can read it instead of blocking the page as a CORS error.
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Backend proxy error' }));
    return;
  }

  // Copy the upstream status and headers back, then stream the body.
  res.writeHead(upstream.status, Object.fromEntries(upstream.headers.entries()));
  const reader = upstream.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
    }
  } catch (err) {
    // The client went away early; there is nothing useful to return.
  } finally {
    res.end();
  }
};