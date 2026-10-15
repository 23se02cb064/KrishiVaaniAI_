// Deterministic end-to-end tests for the proxy timeout behaviour.
//
// No real network and no 120s waits: a local mock backend serves gzipped JSON,
// plain 422 JSON, and a never-responding endpoint. The real proxy handler is
// imported and driven through a real http server -- the same contract
// Vercel's runtime uses. This proves:
//   1. a quick upstream response is forwarded correctly,
//   2. a request exceeding the timeout yields a controlled 502,
//   3. gzip responses remain valid after the fix,
//   4. HTTP 422 responses are forwarded unchanged.
//
// The proxy reads RENDER_BACKEND_URL at import time, so we start a mock
// backend first, point the env at it, then import the handler.

import http from "node:http";
import zlib from "node:zlib";

// 1) Start a mock "Render" backend on a random port.
const payload = JSON.stringify({ predictions: [{ class_name: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot" }] });
const gz = zlib.gzipSync(Buffer.from(payload));

const backend = http.createServer((req, res) => {
  // A slow endpoint that never responds in time, used to trigger the timeout.
  if (req.url.startsWith("/api/slow")) {
    // Never respond; the proxy's AbortSignal will fire first.
    return;
  }
  if (req.method === "POST" && req.url.startsWith("/api/analyze")) {
    // Small, unencoded 422 body -- exactly the case that used to pass through
    // the buggy proxy while gzipped responses failed.
    const body = JSON.stringify({
      error: "Please upload or capture a clear plant leaf image only.",
      error_type: "NON_LEAF_IMAGE",
      leaf_gate: { is_leaf: false, state: "NON_LEAF" },
    });
    res.writeHead(422, { "content-type": "application/json" });
    res.end(body);
    return;
  }
  res.writeHead(200, {
    "content-type": "application/json",
    "content-encoding": "gzip",
    "content-length": String(gz.length),
    "access-control-allow-origin": "*",
  });
  res.end(gz);
});

await new Promise((r) => backend.listen(0, "127.0.0.1", r));
const port = backend.address().port;
process.env.RENDER_BACKEND_URL = "http://127.0.0.1:" + port;

// 2) Now import the real deployed handler.
const { default: proxy } = await import("../frontend/api/proxy.js");

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log("PASS: " + name); }
  else { console.log("FAIL: " + name + (detail ? " - " + detail : "")); failures++; }
}

// Drive the proxy through a real http server the way Vercel does.
function callProxy(path, opts = {}) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      proxy(req, res).catch(reject);
    });
    server.listen(0, "127.0.0.1", async () => {
      const p = server.address().port;
      try {
        const r = await fetch("http://127.0.0.1:" + p + path, opts);
        const body = Buffer.from(await r.arrayBuffer());
        const headers = Object.fromEntries(r.headers.entries());
        server.close(() => resolve({ status: r.status, headers, body }));
      } catch (e) { server.close(() => reject(e)); }
    });
  });
}

// 3) Quick upstream: forwarded correctly, gzip valid.
console.log("--- 1. quick upstream ---");
const r1 = await callProxy("/api/predictions?limit=3");
check("status 200", r1.status === 200, "got " + r1.status);
check("no stale content-encoding", !r1.headers["content-encoding"], "got " + r1.headers["content-encoding"]);
let j1 = null;
try { j1 = JSON.parse(r1.body.toString()); } catch (e) { check("body parses as JSON", false, e.message); }
check("body parses as JSON", !!j1 && Array.isArray(j1.predictions), "no predictions array");

// 4) Upstream exceeds the timeout: controlled 502. We override
//    AbortSignal.timeout with a 30ms version so the test does not wait 120s,
//    and point the request at a mock endpoint that never responds.
console.log("--- 2. upstream exceeds timeout ---");
const realTimeout = AbortSignal.timeout;
let abortFired = false;
AbortSignal.timeout = function (ms) {
  const c = new AbortController();
  setTimeout(() => { abortFired = true; c.abort(); }, 30);
  return c.signal;
};
try {
  const r2 = await callProxy("/api/slow", { method: "GET" });
  check("status 502 on timeout", r2.status === 502, "got " + r2.status);
  try { JSON.parse(r2.body.toString()); check("502 body is JSON", true); }
  catch (e) { check("502 body is JSON", false, r2.body.toString().slice(0, 80)); }
  check("502 carries CORS", r2.headers["access-control-allow-origin"] === "*", "got " + r2.headers["access-control-allow-origin"]);
  check("abort actually fired", abortFired, "timeout did not trigger");
} finally { AbortSignal.timeout = realTimeout; }

// 5) gzip responses remain valid after the fix.
console.log("--- 3. gzip validity ---");
const r3 = await callProxy("/api/predictions?limit=3");
check("status 200", r3.status === 200, "got " + r3.status);
check("no stale content-encoding", !r3.headers["content-encoding"], "got " + r3.headers["content-encoding"]);
check("content-type preserved", r3.headers["content-type"] === "application/json", "got " + r3.headers["content-type"]);
let j3 = null;
try { j3 = JSON.parse(r3.body.toString()); } catch (e) { check("body parses as JSON", false, e.message); }
check("body parses as JSON", !!j3 && Array.isArray(j3.predictions), "no predictions array");

// 6) HTTP 422 forwarded unchanged.
console.log("--- 4. HTTP 422 forwarding ---");
const form = new FormData();
form.append("image", new Blob(["x"], { type: "image/jpeg" }), "leaf.jpg");
form.append("lang", "en");
const r4 = await callProxy("/api/analyze", { method: "POST", body: form });
check("status 422", r4.status === 422, "got " + r4.status);
check("no stale content-encoding on 422", !r4.headers["content-encoding"], "got " + r4.headers["content-encoding"]);
let j4 = null;
try { j4 = JSON.parse(r4.body.toString()); } catch (e) { check("422 body parses as JSON", false, e.message); }
check("422 body parses as JSON", !!j4 && j4.error_type === "NON_LEAF_IMAGE", "got " + JSON.stringify(j4).slice(0, 80));

backend.closeAllConnections?.();
backend.close();
// Give the server a tick to drain open sockets (the /api/slow timeout test
// leaves one open until the abort fires) before exiting, so Node does not
// trip the UV_HANDLE_CLOSING assertion on Windows.
await new Promise((r) => setTimeout(r, 100));
process.exit(failures === 0 ? 0 : 1);