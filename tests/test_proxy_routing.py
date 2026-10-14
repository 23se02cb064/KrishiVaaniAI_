"""Regression tests for the Vercel proxy routing and API base URL.

These guard the production wiring:
  1. The frontend must call relative /api/... URLs, never a hardcoded Render
     host, so every request goes through the Vercel same-origin proxy.
  2. The proxy must forward query strings, methods, and response bodies
     unchanged, and must carry CORS headers on its own error responses.

They are source-level tests (no network, no deploy): they parse the frontend
source and the proxy module and assert on the wiring that matters.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

API_JS = os.path.join(ROOT, 'frontend', 'src', 'api.js')
PROXY_JS = os.path.join(ROOT, 'api', 'proxy.js')
VERCEL_JSON = os.path.join(ROOT, 'vercel.json')


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


def test_proxy_routes_every_api_path():
    """vercel.json must route /api/* through @vercel/node."""
    cfg = json.load(open(VERCEL_JSON))
    builds = {b['src']: b['use'] for b in cfg.get('builds', [])}
    assert 'api/proxy.js' in builds, (
        'vercel.json must build api/proxy.js with @vercel/node; builds=%s'
        % builds)
    assert builds['api/proxy.js'] == '@vercel/node'
    routes = cfg.get('routes', [])
    api_route = next((r for r in routes if r.get('src') == '/api/(.*)'), None)
    assert api_route is not None, (
        'vercel.json must route /api/(.*) to the proxy; routes=%s' % routes)
    assert api_route['dest'] == '/api/proxy.js'
    spa = next((r for r in routes if r.get('src') == '/(.*)'), None)
    assert spa is not None and spa['dest'] == '/index.html'


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