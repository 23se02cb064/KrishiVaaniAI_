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
    Vercel actually reads. It must build index.html (@vercel/static) and
    api/proxy.js (@vercel/node) and route /api/* through the proxy. It must
    NOT use the unsupported rootDirectory property, which fails schema
    validation. All build sources must exist relative to the frontend dir.
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
    builds = {b['src']: b['use'] for b in cfg.get('builds', [])}
    assert 'api/proxy.js' in builds and builds['api/proxy.js'] == '@vercel/node'
    assert 'index.html' in builds and builds['index.html'] == '@vercel/static'
    routes = cfg.get('routes', [])
    assert any(r.get('src') == '/api/(.*)' and r.get('dest') == '/api/proxy.js'
               for r in routes)
    assert any(r.get('src') == '/(.*)' and r.get('dest') == '/index.html'
               for r in routes)
    for src in builds:
        assert os.path.exists(os.path.join(ROOT, 'frontend', src)), (
            'vercel.json build source %r does not exist under frontend/, so '
            'the build would fail' % src)


def test_proxy_lives_inside_the_root_directory():
    """The deployed proxy must live inside the Vercel root directory
    (frontend/api/proxy.js), because vercel.json is read from there and its
    build sources are resolved relative to it."""
    assert os.path.exists(PROXY_JS), (
        'frontend/api/proxy.js must exist so the @vercel/node build resolves '
        'when Root Directory is frontend')
    src = _read(PROXY_JS)
    assert 'createProxyMiddleware' in src
    assert 'RENDER_BACKEND_URL' in src
    assert 'Access-Control-Allow-Origin' in src


def test_proxy_forwards_path_and_query_string():
    """The proxy must not rewrite the incoming path; it forwards it verbatim."""
    src = _read(PROXY_JS)
    assert 'pathRewrite' in src, 'the proxy uses pathRewrite'
    assert 'return path' in src, (
        'pathRewrite must return the original path unchanged so query strings '
        'and routes are forwarded verbatim to the backend')


def test_proxy_error_path_carries_cors():
    """The proxy must set CORS headers before proxying, so its own error page
    (502/504) is not blocked by the browser as a CORS violation."""
    src = _read(PROXY_JS)
    assert 'applyCors' in src, 'the proxy has a CORS helper'
    assert 'Access-Control-Allow-Origin' in src
    assert 'Access-Control-Allow-Credentials' in src
    assert 'Vary' in src
    # applyCors must run BEFORE the proxy call, not inside the error callback,
    # because http-proxy-middleware writes its own body on failure and sets
    # res.headersSent before the callback is reached.
    assert src.index('applyCors(req, res)') < src.index('apiProxy('), (
        'applyCors must run before the proxy so error responses carry CORS '
        'headers even when http-proxy-middleware writes its own body first')


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