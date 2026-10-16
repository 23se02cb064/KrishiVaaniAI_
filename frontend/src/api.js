const API_BASE = `${import.meta.env.VITE_API_URL || ''}/api`;
async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  const response = await fetch(url, config);
  if (!response.ok) {
    let data = null;
    try {
      data = await response.json();
    } catch {
      // Non-JSON body (e.g. gateway HTML page). Treat as server-side failure.
      data = { error: `Server returned HTTP ${response.status}` };
    }
    const error = new Error(data.error || data.message || 'Request failed');
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return response.json();
}

export const api = {
  analyze: (formData) => {
    const fd = new FormData();
    if (formData.get('lang')) fd.append('lang', formData.get('lang'));
    if (formData.get('include_gradcam')) fd.append('include_gradcam', formData.get('include_gradcam'));
    const file = formData.get('image');
    if (file) fd.append('image', file);
    return fetch(`${API_BASE}/analyze`, { method: 'POST', body: fd });
  },

  getTrainingStatus: () => request('/training/status'),

  startTraining: () => request('/training/start', { method: 'POST' }),

  getPredictions: (limit = 10) => request(`/predictions?limit=${limit}`),

  getPredictionAnalytics: () => request('/predictions/analytics'),

  chat: (query, lang = 'en') =>
    request('/chat', {
      method: 'POST',
      body: JSON.stringify({ query, lang }),
    }),

  weather: (lat, lon) => {
    let url = `${API_BASE}/weather`;
    if (lat && lon) url += `?lat=${lat}&lon=${lon}`;
    return fetch(url).then(async (r) => {
      if (!r.ok) {
        throw new Error(`Weather service returned HTTP ${r.status}`);
      }
      return r.json();
    });
  },

  tts: (text, lang = 'en') =>
    request('/tts', {
      method: 'POST',
      body: JSON.stringify({ text, lang }),
    }),
};
