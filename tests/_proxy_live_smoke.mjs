// Live smoke test: invoke the FIXED proxy.js handler directly against the
// REAL Render backend, proving the gzip fix works end-to-end without
// deploying. The proxy is a Vercel serverless function, so it is imported and
// called with a real http server the way Vercel's runtime calls it.
//
// Usage: RENDER_BACKEND_URL=https://krishivaaniai-ih8d.onrender.com node tests/_proxy_live_smoke.mjs

import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';

const BACKEND = process.env.RENDER_BACKEND_URL;
if (!BACKEND) {
  console.error('RENDER_BACKEND_URL not set');
  process.exit(2);
}

const { default: proxy } = await import('../frontend/api/proxy.js');

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log(`PASS: ${name}`); }
  else { console.log(`FAIL: ${name}${detail ? ' - ' + detail : ''}`); failures++; }
}

function callProxy(path, opts = {}) {
  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      proxy(req, res).catch(reject);
    });
    server.listen(0, '127.0.0.1', async () => {
      const port = server.address().port;
      try {
        const r = await fetch(`http://127.0.0.1:${port}${path}`, opts);
        const body = Buffer.from(await r.arrayBuffer());
        const headers = Object.fromEntries(r.headers.entries());
        server.close(() => resolve({ status: r.status, headers, body }));
      } catch (e) {
        server.close(() => reject(e));
      }
    });
  });
}

// 1) GET /api/predictions through the fixed proxy -- the previously broken path.
console.log('--- GET /api/predictions (fixed proxy, real Render) ---');
let r1;
try {
  r1 = await callProxy('/api/predictions?limit=3', { headers: { 'Accept-Encoding': 'gzip' } });
} catch (e) { console.log('proxy call failed:', e.message); process.exit(1); }
console.log('  status:', r1.status, 'content-encoding:', r1.headers['content-encoding'], 'len:', r1.body.length, 'first4:', Buffer.from(r1.body.slice(0,4)).toString('hex'));
check('proxy returns 200', r1.status === 200, `got ${r1.status}`);
check('proxy strips stale content-encoding', !r1.headers['content-encoding'], `got ${r1.headers['content-encoding']}`);
let j1 = null;
try { j1 = JSON.parse(r1.body.toString()); } catch (e) { check('body parses as JSON', false, e.message); }
check('body parses as JSON', !!j1 && Array.isArray(j1.predictions), 'no predictions array');

// 2) POST /api/analyze with a real leaf image through the fixed proxy.
const imgPath = 'Plant_Disease-reference/static/000_jpg.rf.79b07e9657b862eca6972264b5e2fe0c.jpg';
let img;
try { img = readFileSync(imgPath); } catch (e) { console.log('SKIP analyze (no test image):', e.message); }
if (img) {
  console.log('--- POST /api/analyze (fixed proxy, real Render) ---');
  const form = new FormData();
  form.append('image', new Blob([img], { type: 'image/jpeg' }), 'leaf.jpg');
  form.append('lang', 'en');
  form.append('include_gradcam', 'false');
  form.append('source', 'upload');
  let r2;
  try {
    r2 = await callProxy('/api/analyze', { method: 'POST', body: form });
  } catch (e) { console.log('proxy call failed:', e.message); process.exit(1); }
  console.log('  status:', r2.status, 'content-encoding:', r2.headers['content-encoding'], 'len:', r2.body.length, 'body:', r2.body.toString().slice(0, 200));
check('analyze returns 200', r2.status === 200, `got ${r2.status}`);
  check('analyze strips stale content-encoding', !r2.headers['content-encoding'], `got ${r2.headers['content-encoding']}`);
  let j2 = null;
  try { j2 = JSON.parse(r2.body.toString()); } catch (e) { check('analyze body parses as JSON', false, e.message + ' body=' + r2.body.toString().slice(0,100)); }
  if (j2) {
    check('analyze body has a class prediction', typeof j2.class === 'string' && j2.class.length > 0, `got ${JSON.stringify(j2.class)}`);
    check('analyze body has confidence', typeof j2.confidence === 'number', `got ${j2.confidence}`);
    console.log('   class:', j2.class, 'confidence:', j2.confidence);
  } else {
    console.log('   raw body:', r2.body.toString());
  }
}

process.exit(failures === 0 ? 0 : 1);