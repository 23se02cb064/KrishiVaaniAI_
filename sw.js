const CACHE_VERSION = 'ch-ai-v4';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const API_CACHE = `${CACHE_VERSION}-api`;

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/main.js',
  '/manifest.json',
  '/static/images/Screenshot 2025-06-25 121137.png',
  '/static/images/Screenshot 2025-06-24 004009.png',
  '/static/images/pixel-art-retro-videogame-landscape-background-combine-harvester-wheat-field-300530648.webp',
];

const EXTERNAL_URLS = [
  'https://cdn.tailwindcss.com',
  'https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap',
  'https://fonts.gstatic.com/s/pressstart2p/v15/e3t4euO8T-267oIAQAu6jDQyK0nS.woff2',
];

self.addEventListener('install', (event) => {
  console.log('[SW] Install');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Precaching app shell');
        return cache.addAll([...PRECACHE_URLS, ...EXTERNAL_URLS]);
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  console.log('[SW] Activate');
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((key) => key !== STATIC_CACHE && key !== API_CACHE)
          .map((key) => {
            console.log('[SW] Removing old cache:', key);
            return caches.delete(key);
          })
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET') {
    return;
  }

  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }

  if (request.destination === 'image' || request.destination === 'font' || request.destination === 'style' || request.destination === 'script') {
    event.respondWith(handleStaticAsset(request));
    return;
  }

  event.respondWith(handleNavigation(request));
});

async function handleNavigation(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[SW] Navigation failed, serving from cache:', error);
    const cachedResponse = await caches.match('/index.html');
    if (cachedResponse) {
      return cachedResponse;
    }
    return new Response('Offline - App not available', { status: 503, headers: { 'Content-Type': 'text/html' } });
  }
}

async function handleStaticAsset(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[SW] Asset fetch failed:', request.url);
    return new Response('Offline', { status: 503 });
  }
}

async function handleApiRequest(request) {
  const url = new URL(request.url);
  const cacheKey = `${request.method}-${url.pathname}${url.search}`;

  if (request.method === 'GET') {
    const cachedResponse = await caches.match(cacheKey);
    if (cachedResponse) {
      return cachedResponse;
    }

    try {
      const networkResponse = await fetch(request);
      if (networkResponse && networkResponse.status === 200) {
        const cache = await caches.open(API_CACHE);
        cache.put(cacheKey, networkResponse.clone());
      }
      return networkResponse;
    } catch (error) {
      console.log('[SW] API fetch failed, serving cache:', error);
      const cachedResponse = await caches.match(cacheKey);
      if (cachedResponse) {
        return cachedResponse;
      }
      return createOfflineApiResponse(url.pathname);
    }
  }

  return fetch(request);
}

async function refreshCache(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
  } catch (error) {
    console.log('[SW] Background refresh failed:', error);
  }
}

async function refreshApiCache(request, cacheKey) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      const cache = await caches.open(API_CACHE);
      cache.put(cacheKey, networkResponse.clone());
    }
  } catch (error) {
    console.log('[SW] Background API refresh failed:', error);
  }
}

function createOfflineApiResponse(pathname) {
  let body = { error: 'Offline' };
  let status = 503;

  if (pathname === '/api/weather') {
    body = { weather: 'Offline - cached data unavailable' };
  } else if (pathname === '/api/chat') {
    body = { reply: 'Offline - please check your connection and try again' };
  } else if (pathname === '/api/analyze') {
    body = { error: 'Offline - image analysis requires internet connection' };
  }

  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-pending-requests') {
    event.waitUntil(syncPendingRequests());
  }
});

async function syncPendingRequests() {
  console.log('[SW] Background sync triggered');

  const db = await openDB();
  const tx = db.transaction('pending-requests', 'readwrite');
  const store = tx.objectStore('pending-requests');
  const requests = await store.getAll();

  for (const req of requests) {
    try {
      const response = await fetch(req.url, {
        method: req.method,
        headers: req.headers,
        body: req.body,
      });

      if (response.ok || response.status === 400 || response.status === 401 || response.status === 403 || response.status === 422) {
        await store.delete(req.id);
        console.log('[SW] Synced and removed request:', req.id);

        if ('clients' in self) {
          const allClients = await self.clients.matchAll();
          allClients.forEach(client => {
            client.postMessage({
              type: 'SYNC_COMPLETE',
              requestId: req.id,
              success: response.ok,
            });
          });
        }
      }
    } catch (error) {
      console.log('[SW] Sync failed for request', req.id, ':', error);
    }
  }
}

self.addEventListener('push', (event) => {
  if (!event.data) {
    return;
  }

  let payload = {};
  try {
    payload = event.data.json();
  } catch (e) {
    payload = { title: 'CH AI', body: event.data.text() };
  }

  const options = {
    body: payload.body || 'New notification',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/icon-192x192.png',
    data: payload.data || {},
    actions: payload.actions || [],
    requireInteraction: true,
    tag: payload.tag || 'ch-ai-notification',
    renotify: true,
  };

  event.waitUntil(
    self.registration.showNotification(payload.title || 'CH AI', options)
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow('/')
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('CHAI_OfflineDB', 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending-requests')) {
        db.createObjectStore('pending-requests', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('offline-history')) {
        const historyStore = db.createObjectStore('offline-history', { keyPath: 'id', autoIncrement: true });
        historyStore.createIndex('timestamp', 'timestamp');
      }
      if (!db.objectStoreNames.contains('cached-responses')) {
        db.createObjectStore('cached-responses', { keyPath: 'key' });
      }
    };
  });
}
