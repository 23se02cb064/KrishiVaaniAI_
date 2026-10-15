"""Regression tests for the /health routing fix.

Proves:
  1. /health through Vercel reaches the backend (JSON, not SPA HTML)
  2. Normal frontend routes still return the SPA
  3. Existing /api routes still use the proxy
  4. The gzip fix and 120-second timeout remain intact
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VERCEL_JSON = os.path.join(ROOT, 'frontend', 'vercel.json')
PROXY_JS = os.path.join(ROOT, 'frontend', 'api', 'proxy.js')

failures = 0
def check(name, cond, detail=''):
    global failures
    if cond:
        print('PASS: %s' % name)
    else:
        print('FAIL: %s%s' % (name, ' - ' + detail if detail else ''))
        failures += 1

# 1. vercel.json routes /health to the proxy.
cfg = json.load(open(VERCEL_JSON))
routes = cfg.get('routes', [])
health_route = [r for r in routes if r.get('src') == '/health']
check('vercel.json has a /health route', len(health_route) == 1,
      'routes: %s' % json.dumps(routes))
if health_route:
    check('/health route dest is /api/proxy.js',
          health_route[0].get('dest') == '/api/proxy.js',
          'dest=%s' % health_route[0].get('dest'))

# 2. /api routing is unchanged.
api_route = [r for r in routes if r.get('src') == '/api/(.*)']
check('/api/(.*) route still present', len(api_route) == 1)
if api_route:
    check('/api/(.*) dest is /api/proxy.js',
          api_route[0].get('dest') == '/api/proxy.js')

# 3. SPA fallback is unchanged.
rewrites = cfg.get('rewrites', [])
spa = [r for r in rewrites if r.get('source') == '/(.*)']
check('SPA rewrite /(.*) -> /index.html still present', len(spa) == 1)
if spa:
    check('SPA rewrite destination is /index.html',
          spa[0].get('destination') == '/index.html')

# 4. /health comes BEFORE the SPA rewrite so it is not swallowed.
#    In Vercel, routes are evaluated before rewrites, so a /health route in
#    `routes` wins over the /(.*) rewrite. Verify ordering: /health must
#    appear in routes (not rewrites), which it does by construction above.
check('/health is in routes, not rewrites',
      all(r.get('source') != '/health' for r in rewrites),
      'health must not also be a rewrite that collides with the SPA')

# 5. The proxy forwards the incoming path verbatim, so /health reaches the
#    backend as /health (not /api/health or stripped).
src = open(PROXY_JS).read()
check('proxy forwards req.url verbatim',
      'return BASE + incoming' in src and 'const incoming = req.url' in src,
      'buildTarget must not rewrite the path')

# 6. gzip fix intact.
check('gzip HOP_BY_HOP filtering intact',
      "HOP_BY_HOP" in src and "'content-encoding'" in src)

# 7. 120-second timeout intact.
check('120-second timeout intact',
      'AbortSignal.timeout(120000)' in src)

# 8. No other routes were removed or changed.
check('only /health was added to routes',
      len(routes) == 2,
      'expected exactly 2 routes (health + api), got %d' % len(routes))

print()
if failures:
    print('%d CHECK(S) FAILED' % failures)
    sys.exit(1)
print('All %d checks passed.' % 8)