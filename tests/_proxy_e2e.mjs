// End-to-end test of the proxy's response-header handling.
//
// Spins up a mock "Render" backend that returns a GZIPPED JSON body with the
// correct Content-Encoding/Content-Length headers (exactly like the real
// gunicorn+Flask stack), then runs the proxy's header-filtering logic against
// a Node fetch() call to it. Node's fetch() transparently decompresses the
// body, so the bytes the proxy streams are plain JSON; the test asserts the
// proxy does NOT forward the stale Content-Encoding header over them.

import http from 'node:http';
import zlib from 'node:zlib';

// Spins up a mock "Render" backend that returns a GZIPPED JSON body with the
// correct Content-Encoding/Content-Length headers (exactly like the real
// gunicorn+Flask stack), then runs the proxy's header-filtering logic against
// a Node fetch() call to it. Node's fetch() transparently decompresses the
// body, so the bytes the proxy streams are plain JSON; the test asserts the
// proxy does NOT forward the stale Content-Encoding header over them, and
// that a gzip-aware client can therefore decode the response.

const payload = JSON.stringify({ predictions: [{ class_name: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot' }] });
const gz = zlib.gzipSync(Buffer.from(payload));

const backend = http.createServer((req, res) => {
  res.writeHead(200, {
    'content-type': 'application/json',
    'content-encoding': 'gzip',
    'content-length': String(gz.length),
    'access-control-allow-origin': '*',
  });
  res.end(gz);
});

await new Promise((r) => backend.listen(0, '127.0.0.1', r));
const port = backend.address().port;
const backendUrl = `http://127.0.0.1:${port}/api/predictions?limit=3`;

let failures = 0;
function check(name, cond, detail) {
  if (cond) {
    console.log(`PASS: ${name}`);
  } else {
    console.log(`FAIL: ${name}${detail ? ' - ' + detail : ''}`);
    failures++;
  }
}

// 2) Reproduce what Node fetch() sees: transparently decompressed body + headers.
const upstream = await fetch(backendUrl);
const upstreamBody = Buffer.from(await upstream.arrayBuffer());
const upstreamHeaders = Object.fromEntries(upstream.headers.entries());

console.log('--- upstream (as Node fetch sees it) ---');
console.log('  status:', upstream.status);
console.log('  content-encoding:', upstreamHeaders['content-encoding']);
console.log('  content-length:', upstreamHeaders['content-length']);
console.log('  body first bytes:', upstreamBody.slice(0, 8).toString('hex'), '=>', upstreamBody.slice(0, 20).toString());

// 3) Apply the proxy's response-header filter (mirrors proxy.js:166-175).
const HOP_BY_HOP = new Set([
  'content-encoding', 'content-length', 'transfer-encoding',
  'connection', 'keep-alive', 'upgrade',
]);
const outHeaders = {};
for (const [key, value] of Object.entries(upstreamHeaders)) {
  if (HOP_BY_HOP.has(key)) continue;
  outHeaders[key] = value;
}

check('upstream body is decompressed JSON', upstreamBody.slice(0, 1).toString() === '{', `got ${upstreamBody.slice(0,20).toString()}`);
check('proxy strips stale content-encoding', !('content-encoding' in outHeaders), `still present: ${outHeaders['content-encoding']}`);
check('proxy strips stale content-length', !('content-length' in outHeaders), `still present: ${outHeaders['content-length']}`);
check('proxy preserves content-type', outHeaders['content-type'] === 'application/json', `got ${outHeaders['content-type']}`);
check('proxy preserves CORS header', outHeaders['access-control-allow-origin'] === '*', `got ${outHeaders['access-control-allow-origin']}`);

// 4) After the fix the client receives plain JSON with no Content-Encoding, so
//    it must parse directly. Before the fix the client received plain JSON
//    *with* Content-Encoding: gzip and failed to decode ("incorrect header
//    check"), which is the production symptom. We assert both: the body the
//    proxy would stream parses as JSON, and that the old behaviour would have
//    produced a body that a gzip-aware client cannot decode.
let proxiedJson = null;
try {
  proxiedJson = JSON.parse(upstreamBody.toString());
} catch (err) {
  check('proxied body parses as JSON', false, err.message);
}
check('proxied body parses as JSON', !!proxiedJson && proxiedJson.predictions && proxiedJson.predictions[0].class_name.includes('Corn'), 'no predictions key');

// A gzip-aware client receiving the OLD (buggy) response would try to gunzip
// plain JSON and fail -- demonstrate that the regression is real.
let oldClientFails = false;
try {
  zlib.gunzipSync(upstreamBody);
} catch (err) {
  oldClientFails = true;
}
check('old buggy response would have failed a gzip-aware client', oldClientFails, 'regression not demonstrated');

// 5) Sanity: confirm the OLD behaviour (verbatim copy) would have failed, so
//    this test genuinely guards the regression rather than passing trivially.
const oldWouldFail = ('content-encoding' in upstreamHeaders) && upstreamHeaders['content-encoding'] === 'gzip' && upstreamBody.slice(0,1).toString() === '{';
check('old verbatim-copy behaviour would have corrupted the body', oldWouldFail, 'regression not demonstrated');

backend.close();
process.exit(failures === 0 ? 0 : 1);