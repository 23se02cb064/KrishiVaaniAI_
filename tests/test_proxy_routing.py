"""Regression tests for the Vercel proxy routing and API base URL.

These guard the production wiring:
  1. The frontend must call relative /api/... URLs, never a hardcoded Render
     host, so every request goes through the Vercel same-origin proxy.
  2. The proxy must live INSIDE the Vercel root directory
     (frontend/api/proxy.js), because vercel.json is read from there and its
     build sources are resolved relative to it. The unsupported
     rootDirectory property must never appear in vercel.json.
  3. The proxy must forward query strings, methods, and response bodies
     unchanged, and must carry CORS headers on its own error responses.

They are source-level tests (no network, no deploy): they parse the frontend
source, the proxy module, and the vercel.json files and assert on the wiring
that matters.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

API_JS = os.path.join(ROOT, 'frontend', 'src', 'api.js')
PROXY_JS = os.path.join(ROOT, 'frontend', 'api', 'proxy.js')
FRONTEND_VERCEL_JSON = os.path.join(ROOT, 'frontend', 'vercel.json')
ROOT_VERCEL_JSON = os.path.join(ROOT, 'vercel.json')


def _read(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return fh.read()


def test_api_base_url_is_relative():
    """Every frontend API call must be relative to /api, never a hardcoded host."""
    src = _read(API_JS)
    assert 'VITE_API_URL' in src, 'the frontend must read VITE_API_URL'
    assert "import.meta.env.VITE_API_URL || '" in src, (
        'the frontend must fall back to an empty string, so a missing env var '
        'yields a relative /api URL rather than a broken absolute one')
    for bad in ('krishivaaniai-ih8d.onrender.com', 'https://render.com',
                'RENDER_BACKEND_URL'):
        assert bad not in src, (
            'frontend request code must not hard-code %r; it must use the '
            'relative /api path through the Vercel proxy' % bad)


def test_api_base_url_has_no_default_render_host():
    """The fallback must be empty, never a Render URL."""
    src = _read(API_JS)
    match = re.search(r"VITE_API_URL \|\| '([^']*)'", src)
    assert match is not None, 'could not find the VITE_API_URL fallback'
    assert match.group(1) == '', (
        'the VITE_API_URL fallback must be an empty string, got %r'
        % match.group(1))


def test_frontend_vercel_json_is_valid_and_routes_api():
    """With Vercel Root Directory = frontend, frontend/vercel.json is the file
    Vercel actually reads. It must declare a build command that produces the
    full dist/ directory (so JS/CSS/images are published), route /api/* through
    the proxy, and serve the SPA on every other path. It must NOT use the
    unsupported rootDirectory property, which fails schema validation.
    """
    path = FRONTEND_VERCEL_JSON
    assert os.path.exists(path), (
        'frontend/vercel.json must exist: with Root Directory=frontend, Vercel '
        'ignores the repo-root vercel.json and never builds api/proxy.js')
    cfg = json.load(open(path))
    assert 'rootDirectory' not in cfg, (
        'vercel.json schema forbids rootDirectory; it is a project-level '
        'setting, not a config property. Put api/proxy.js inside the root '
        'directory instead.')
    assert cfg.get('buildCommand') == 'npm run build', (
        'vercel.json must declare npm run build so Vite produces the full '
        'dist/ directory (index.html + assets/) that Vercel publishes')
    assert 'builds' not in cfg, (
        'vercel.json must NOT use the legacy builds array with @vercel/static: '
        '@vercel/static copies only the single entry file it is given, so the '
        'app\'s JS and CSS in dist/assets/ would be unreachable (404) at '
        'runtime. Use Vercel\'s native Vite framework integration instead.')
    routes = cfg.get('routes', [])
    assert any(r.get('src') == '/api/(.*)' and r.get('dest') == '/api/proxy.js'
               for r in routes), (
        'vercel.json must route /api/* through the proxy verbatim')
    # The SPA fallback must NOT be in routes: a catch-all in routes would
    # rewrite /assets/*.js and /assets/*.css to index.html before the
    # filesystem is checked, leaving the app a 404 shell. Vercel's documented
    # SPA pattern puts it in rewrites instead, which checks the filesystem
    # first and only rewrites paths that have no static file.
    assert not any(r.get('src') == '/(.*)' for r in routes), (
        'vercel.json must NOT put the catch-all in routes: it would rewrite '
        'static assets (dist/assets/*.js, *.css) to index.html before the '
        'filesystem is checked, so the SPA would be a 404 shell. Use rewrites '
        'instead, which checks the filesystem first.')
    rewrites = cfg.get('rewrites', [])
    assert any(r.get('source') == '/(.*)' and r.get('destination') == '/index.html'
               for r in rewrites), (
        'vercel.json must rewrite /(.*) to /index.html in the rewrites array, '
        'which checks the filesystem first and only rewrites paths that have '
        'no static file')
    # The build must actually produce the assets the SPA needs.
    dist = os.path.join(ROOT, 'frontend', 'dist')
    assert os.path.exists(os.path.join(dist, 'index.html')), (
        'npm run build must produce dist/index.html')
    assets = os.path.join(dist, 'assets')
    assert os.path.isdir(assets) and os.listdir(assets), (
        'npm run build must produce dist/assets/ with the app\'s JS and CSS, '
        'otherwise the SPA is a 404 shell')
    # The proxy must exist and be ESM (export default), because Vercel copies
    # the root package.json (type:module) into every function directory.
    proxy_src = os.path.join(ROOT, 'frontend', 'api', 'proxy.js')
    assert os.path.exists(proxy_src), 'frontend/api/proxy.js must exist'
    psrc = _read(proxy_src)
    assert 'export default' in psrc, (
        'the proxy must be ESM (export default): Vercel copies the root '
        'package.json (which carries "type": "module") into every function '
        'directory, and under "type": "module" a CommonJS module.exports is '
        'silently ignored, so a CJS proxy would export nothing and every '
        '/api/* request would 500')
    assert 'fetch(' in psrc, 'the proxy must use Node\'s built-in fetch()'


def test_proxy_lives_inside_the_root_directory():
    """The deployed proxy must live inside the Vercel root directory
    (frontend/api/proxy.js), because vercel.json is read from there and its
    build sources are resolved relative to it."""
    assert os.path.exists(PROXY_JS), (
        'frontend/api/proxy.js must exist so the @vercel/node build resolves '
        'when Root Directory is frontend')
    src = _read(PROXY_JS)
    assert 'createProxyMiddleware' not in src, (
        'the proxy must not depend on http-proxy-middleware: its ESM-only '
        'build cannot be require()d by Vercel\'s Node runtime (ERR_REQUIRE_ESM)')
    assert 'fetch(' in src, 'the proxy must use Node\'s built-in fetch()'
    assert 'RENDER_BACKEND_URL' in src
    assert 'Access-Control-Allow-Origin' in src


def test_proxy_forwards_path_and_query_string():
    """The proxy must forward the incoming path and query string verbatim to
    the backend, so routes and query parameters survive the hop."""
    src = _read(PROXY_JS)
    assert 'req.url' in src, 'the proxy forwards the incoming URL verbatim'
    assert 'BASE + incoming' in src, (
        'the proxy must concatenate the backend base with the incoming path '
        'and query string unchanged')


def test_proxy_error_path_carries_cors():
    """The proxy must apply CORS before the backend call, so its own error
    response (502) is not blocked by the browser as a CORS violation."""
    src = _read(PROXY_JS)
    assert 'applyCors' in src, 'the proxy has a CORS helper'
    assert 'Access-Control-Allow-Origin' in src
    assert 'Access-Control-Allow-Credentials' in src
    assert 'Vary' in src
    # applyCors must run before the backend fetch() call, because the proxy's
    # own error response is what the browser sees when the backend is
    # unreachable. Compare the LAST occurrence of applyCors against the LAST
    # fetch( call: in the exported async handler, applyCors is called first.
    assert src.rindex('applyCors(req, res)') < src.rindex('fetch('), (
        'applyCors must run before the backend call so the 502 error page '
        'carries CORS headers')


def test_proxy_requires_backend_url():
    """The proxy must fail fast if RENDER_BACKEND_URL is not configured."""
    src = _read(PROXY_JS)
    assert 'RENDER_BACKEND_URL' in src
    assert ("throw new Error('RENDER_BACKEND_URL environment variable "
            "is not configured')") in src


def test_no_unsupported_vercel_json_property():
    """Neither vercel.json file may use a property the schema rejects."""
    for path in (FRONTEND_VERCEL_JSON, ROOT_VERCEL_JSON):
        if not os.path.exists(path):
            continue
        cfg = json.load(open(path))
        for forbidden in ('rootDirectory', 'rootDirectory', 'rootDir'):
            assert forbidden not in cfg, (
                '%s: vercel.json schema rejects %r' % (path, forbidden))