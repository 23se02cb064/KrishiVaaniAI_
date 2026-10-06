// main.js for CH AI
// Handles UI interactions, voice/text input, and API calls

// Elements
const imageInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const textQuery = document.getElementById('textQuery');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const resultsDiv = document.getElementById('results');
const speakAdviceBtn = document.getElementById('speakAdviceBtn');
const weatherDiv = document.getElementById('weather');
const fetchWeatherBtn = document.getElementById('fetchWeatherBtn');
const langSelect = document.getElementById('langSelect');
const stopBtn = document.getElementById('stopBtn');

// Location and Language State
let userLat = null;
let userLon = null;
let userLang = 'en';
let recognitionGlobal = null; // for voice stop control

// i18n translations
const translations = {
  as: {uploadLabel:'পাত বিলাক আপল’ড কৰক', analyzeBtn:'বিশ্লেষণ', askSpeak:'আপোনাৰ প্ৰশ্ন সোধক বা কোৱা', queryPlaceholder:'আপোনাৰ প্ৰশ্ন টাইপ কৰক...', sendBtn:'পঠিয়াওক', speakBtn:'কোৱা', stopBtn:'বন্ধ', listenBtn:'শুনক', weatherHeading:'বতাহ', useLocationBtn:'📍 মোৰ অৱস্থান'},
  gu: {uploadLabel:'પાંદાની ફોટો અપલોડ કરો', analyzeBtn:'વિશ્લેષણ', askSpeak:'તમારો પ્રશ્ન લખો કે બોલો', queryPlaceholder:'તમારો પ્રશ્ન લખો...', sendBtn:'મોકલવું', speakBtn:'બધારવું', stopBtn:'થંભાવો', listenBtn:'સાંભળો', weatherHeading:'હવામান', useLocationBtn:'📍 મારી જગ્યા'},
  kn: {uploadLabel:'ಎಲೆ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ', analyzeBtn:'ವಿಶ್ಲೇಷಿಸು', askSpeak:'ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ ಅಥವಾ ಮಾತನಾಡಿ', queryPlaceholder:'ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ...', sendBtn:'ಕಳುಹಿಸಿ', speakBtn:'ಮಾತನಾಡಿ', stopBtn:'ನಿಲ್ಲಿಸಿ', listenBtn:'ಕೆಳಿ', weatherHeading:'ಹವಾಮಾನ', useLocationBtn:'📍 ನನ್ನ ಸ್ಥಾನ'},
  ml: {uploadLabel:'ഇല ഫോട്ടോ അപ്‌ലോഡ് ചെയ്യുക', analyzeBtn:'വിശകലനം', askSpeak:'ചോദ്യമൈ പൊലി അഡോ', queryPlaceholder:'ചോദ്യം ടൈപ്പ് ചെയ്യുക...', sendBtn:'യയക്കുക', speakBtn:'പറയുക', stopBtn:'നിര്‍ത്തുക', listenBtn:'കേള്‍ക്കുക', weatherHeading:'വാതാവസം', useLocationBtn:'📍 എന്റെ സ്ഥലം'},
  mr: {uploadLabel:'पानाचा फोटो अपलोड करा', analyzeBtn:'विश्लેષण', askSpeak:'तुमचा प्रश्न टाइप करा किंवा बोला', queryPlaceholder:'तुमचा प्रश्न टाइप करा...', sendBtn:'पाठवा', speakBtn:'बोला', stopBtn:'थांबा', listenBtn:'ऐका', weatherHeading:'हवामान', useLocationBtn:'📍 माझे स्थान'},
  pa: {uploadLabel:'ਪੱਤੇ ਦੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ', analyzeBtn:'ਵਿਸ਼ਲੇਸ਼ਣ', askSpeak:'ਆਪਣਾ ਪ੍ਰਸ਼ਨ ਲਿਖੋ ਜਾਂ ਬੋਲੋ', queryPlaceholder:'ਆਪਣਾ ਪ੍ਰਸ਼ਨ ਲਿਖੋ...', sendBtn:'ਭੇਜੋ', speakBtn:'ਬੋਲੋ', stopBtn:'ਰੋਕੋ', listenBtn:'ਸੁਣੋ', weatherHeading:'ਮੌਸਮ', useLocationBtn:'📍 ਮੇਰਾ ਸਥਾਨ'},
  te: {uploadLabel:'ఆకు ఫోటోను అప్‌లోడ్ చేయండి', analyzeBtn:'విశ్లేషణ', askSpeak:'మీ ప్రశ్నను టైప్ చేయండి లేదా మాట్లాడండి', queryPlaceholder:'మీ ప్రశ్నను టైప్ చేయండి...', sendBtn:'పంపండి', speakBtn:'మాట్లాడు', stopBtn:'ఆపు', listenBtn:'విను', weatherHeading:'వాతావరణం', useLocationBtn:'📍 నా స్థానం'},
  ur: {uploadLabel:'پتے کی تصویر اپ لوڈ کریں', analyzeBtn:'تجزیہ', askSpeak:'اپنا سوال لکھیں یا بولیں', queryPlaceholder:'اپنا سوال لکھیں...', sendBtn:'بھیجیں', speakBtn:'بولیں', stopBtn:'روکیں', listenBtn:'سنیں', weatherHeading:'موسم', useLocationBtn:'📍 میری جگہ'},
  or: {uploadLabel:'ପତ୍ର ଫୋଟୋ ଅପଲୋଡ୍ କରନ୍ତୁ', analyzeBtn:'bisleshana', askSpeak:'ଆପଣଙ୍କ ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ କିମ୍ବା କୁହନ୍ତୁ', queryPlaceholder:'ପ୍ରଶ୍ନ ଟାଇପ୍ କରନ୍ତୁ...', sendBtn:'ପଠାନ୍ତୁ', speakBtn:'କହନ୍ତୁ', stopBtn:'ଥାଆନ୍ତୁ', listenBtn:'ଶୁଣନ୍ତୁ', weatherHeading:'ପାଣିପାଗ', useLocationBtn:'📍 ମୋ ଅବସ୍ଥାନ'},
  hi: {uploadLabel:'पत्ते की फोटो अपलोड करें', analyzeBtn:'विश्लेषण', askSpeak:'अपना प्रश्न लिखें या बोलें', queryPlaceholder:'अपना प्रश्न लिखें...', sendBtn:'भेजें', speakBtn:'बोलें', stopBtn:'रोकें', listenBtn:'सुनें', weatherHeading:'मौसम', useLocationBtn:'📍 मेरा स्थान'},
  bn: {uploadLabel:'পাতার ছবি আপলোড করুন', analyzeBtn:'বিশ্লেষণ', askSpeak:'আপনার প্রশ্ন লিখুন বা বলুন', queryPlaceholder:'আপনার প্রশ্ন লিখুন...', sendBtn:'পাঠান', speakBtn:'বলুন', stopBtn:'থামুন', listenBtn:'শুনুন', weatherHeading:'আবহাওয়া', useLocationBtn:'📍 আমার অবস্থান'},
  ta: {uploadLabel:'இலை புகைப்படம் பதிவேற்று', analyzeBtn:'ஆய்வு', askSpeak:'உங்கள் கேள்வியை எழுதுக அல்லது பேசுக', queryPlaceholder:'உங்கள் கேள்வியை எழுதுக...', sendBtn:'அனுப்பு', speakBtn:'பேசு', stopBtn:'நிறுத்து', listenBtn:'கேள்', weatherHeading:'வானிலை', useLocationBtn:'📍 இடம்'},
  en: {uploadLabel:'Upload Crop Leaf Photo', analyzeBtn:'Analyze', askSpeak:'Ask or Speak Your Query', queryPlaceholder:'Type your question...', sendBtn:'Send', speakBtn:'Speak', stopBtn:'Stop', listenBtn:'Listen', weatherHeading:'Weather', useLocationBtn:'📍 Use My Location'}
};

// Speech Synthesis Voices Loader
let speechVoices = [];
function populateVoices() {
  const avail = window.speechSynthesis.getVoices();
  if (avail && avail.length) {
    speechVoices = avail;
  }
}

// Speak helper with local voice or server fallback
async function speakOut(text) {
  if (!text) return;
  try {
    if ('speechSynthesis' in window) {
      populateVoices();
      const lc = userLang.toLowerCase();
      let voice = speechVoices.find(v => v.lang && v.lang.toLowerCase().startsWith(lc));
      if (!voice && lc.includes('-')) {
        const prefix = lc.split('-')[0];
        voice = speechVoices.find(v => v.lang && v.lang.toLowerCase().startsWith(prefix));
      }
      if (voice) {
        const utt = new SpeechSynthesisUtterance(text);
        utt.voice = voice;
        utt.lang = voice.lang;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utt);
        return;
      }
    }
    // fallback to server TTS
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text, lang: userLang})
    });
    const data = await res.json();
    if (data.audio) {
      new Audio('data:audio/mp3;base64,'+data.audio).play();
    } else {
      console.warn('TTS unavailable');
    }
  } catch(e) {
    console.error('TTS error', e);
  }
}

function translateUI() {
  const dict = translations[userLang] || translations['en'];
  document.querySelectorAll('[data-i18n]').forEach(el=>{
    const key = el.getAttribute('data-i18n');
    if(dict[key]) el.textContent = dict[key];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el=>{
    const key = el.getAttribute('data-i18n-placeholder');
    if(dict[key]) el.placeholder = dict[key];
  });
}

// Fetch weather using stored coordinates or geolocation
async function fetchWeather(lat, lon) {
  weatherDiv.textContent = 'Loading weather...';
  try {
    let url = '/api/weather';
    if (lat !== null && lon !== null) {
      url += `?lat=${lat}&lon=${lon}`;
    }
    
    if (!PWA.isOnline()) {
      const cached = await PWA.offlineFetch(url);
      if (cached.fromCache && cached.data) {
        weatherDiv.textContent = cached.data.weather || 'No cached data.';
        return;
      }
      weatherDiv.textContent = 'Offline - no cached data available.';
      return;
    }
    
    const res = await fetch(url);
    const data = await res.json();
    weatherDiv.textContent = data.weather || 'No data.';
  } catch (e) {
    weatherDiv.textContent = 'Error fetching weather.';
  }
}

// Fetch weather with geolocation, updating global coordinates
async function fetchWeatherWithGeo() {
  weatherDiv.textContent = 'Loading weather...';
  if (!('geolocation' in navigator)) {
    weatherDiv.textContent = 'Geolocation not supported.';
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (position) => {
      userLat = position.coords.latitude;
      userLon = position.coords.longitude;
      fetchWeather(userLat, userLon);
    },
    (error) => {
      console.error('Geolocation error:', error);
      if (!PWA.isOnline()) {
        weatherDiv.textContent = 'Offline - location unavailable';
      } else {
        weatherDiv.textContent = 'Location access denied.';
      }
    }
  );
}

// Event Listeners
fetchWeatherBtn.addEventListener('click', fetchWeatherWithGeo);

langSelect.addEventListener('change', function() {
  userLang = langSelect.value;
  translateUI();
});

// Page load handler
window.addEventListener('DOMContentLoaded', () => {
  translateUI();
  fetchWeatherWithGeo();

  // Initialize speech synthesis voices
  if ('speechSynthesis' in window) {
    populateVoices();
    window.speechSynthesis.onvoiceschanged = populateVoices;
  }
});

// Analyze button click
analyzeBtn.addEventListener('click', async () => {
  const file = imageInput.files[0];
  if (!file) {
    resultsDiv.textContent = 'Please upload a leaf image.';
    return;
  }
  const formData = new FormData();
  formData.append('image', file);
  formData.append('lang', userLang);
  formData.append('include_gradcam', 'true');
  resultsDiv.textContent = 'Analyzing...';

  if (!PWA.isOnline()) {
    resultsDiv.textContent = 'Offline: Analysis queued. Will process when connection returns.';
    await PWA.offlineFetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });
    return;
  }

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    // Clear any previous diagnosis state immediately for every response, so a
    // refusal can never leave a stale disease result on screen.
    document.getElementById('resultImages').style.display = 'none';
    document.getElementById('confidenceBar').style.display = 'none';
    const topPredictionsDiv = document.getElementById('topPredictions');
    if (topPredictionsDiv) topPredictionsDiv.style.display = 'none';
    const warningBar = document.getElementById('uncertaintyWarning');
    if (warningBar) warningBar.style.display = 'none';

    // The leaf gate refuses the image before classification with HTTP 422. That is a
    // deliberate "no diagnosis", not a server fault, so it gets a dedicated refusal
    // presentation instead of being reported as an analysis error. Non-leaf and
    // uncertain images must never show a disease result.
    if (!res.ok) {
      if (data && data.error_type) {
        resultsDiv.innerHTML = `
          <div style="padding:10px 12px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;color:#991b1b;">
            <p style="font-weight:600;margin:0 0 6px 0;">
              ${data.error_type === 'LEAF_UNCERTAIN' ? 'Could not confidently identify a leaf' : 'This does not look like a plant leaf image'}
            </p>
            <p style="margin:0;">${data.error}</p>
          </div>
          <div style="margin-top:8px;font-size:0.85rem;color:#4b5563;">
            <p style="font-weight:600;margin:0 0 4px 0;">For a diagnosis, try:</p>
            <ul style="margin:0;padding-left:18px;">
              <li>A close-up of a single leaf, filling most of the frame</li>
              <li>Even lighting, no harsh glare or deep shadow</li>
              <li>Both the whole leaf and any spots or lesions in focus</li>
            </ul>
          </div>
        `;
        return;
      }
      resultsDiv.innerHTML = '<p style="color:#991b1b;">Analysis failed: ' + (data.error || res.statusText) + '</p>';
      return;
    }

    // res.ok === true from here on: this is a genuine diagnosis.
    if (data.class) {
      const confPct = (data.confidence * 100).toFixed(1);
      const parts = data.class.split('___');
      const crop = parts[0] || 'Unknown';
      const disease = parts[1] ? parts[1].replace(/_/g, ' ') : data.class;

      let resultHTML = `
        <p><span style="font-weight:600;">Crop:</span> ${crop}</p>
        <p><span style="font-weight:600;">Disease:</span> ${disease}</p>
        <p><span style="font-weight:600;">Confidence:</span> ${confPct}%</p>
      `;

      if (data.is_uncertain) {
        resultHTML += `
          <div style="margin-top:8px;padding:8px 12px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;font-size:0.85rem;color:#92400e;">
            <strong>Low confidence warning:</strong> The AI is not confident in this prediction.
            Consider consulting a local agricultural expert for a more accurate diagnosis.
          </div>
        `;
      }

      resultHTML += `
        <p style="margin-top:8px;"><span style="font-weight:600;">Recommendation:</span> ${data.remedy || 'Use the recommendation from the disease page.'}</p>
      `;

      resultsDiv.innerHTML = resultHTML;

      // Display top predictions if available
      const topPredictionsDiv = document.getElementById('topPredictions');
      const topPredictionsList = document.getElementById('topPredictionsList');
      if (data.top_k_classes && data.top_k_classes.length > 1) {
        let topHtml = '';
        data.top_k_classes.forEach((pred, i) => {
          const predDisease = pred.class.split('___');
          const predName = predDisease[1] ? predDisease[1].replace(/_/g, ' ') : pred.class;
          topHtml += `<div class="flex justify-between text-xs"><span>${i+1}. ${predName}</span><span>${(pred.confidence*100).toFixed(1)}%</span></div>`;
        });
        topPredictionsList.innerHTML = topHtml;
        topPredictionsDiv.style.display = 'block';
      } else {
        topPredictionsDiv.style.display = 'none';
      }

      // Display uncertainty warning in confidence bar area if low confidence
      if (data.is_uncertain) {
        const warningBar = document.getElementById('uncertaintyWarning');
        if (warningBar) warningBar.style.display = 'block';
      }

      if (data.gradcam_overlay) {
        const originalUrl = URL.createObjectURL(file);
        document.getElementById('originalImage').src = originalUrl;
        document.getElementById('gradcamImage').src = 'data:image/png;base64,' + data.gradcam_overlay;
        document.getElementById('resultImages').style.display = 'block';
      } else {
        document.getElementById('resultImages').style.display = 'none';
      }

      if (data.confidence != null) {
        const pct = (data.confidence * 100).toFixed(1);
        document.getElementById('confidenceFill').style.width = pct + '%';
        document.getElementById('confidenceText').textContent = pct + '%';
        document.getElementById('confidenceBar').style.display = 'block';
      }
    } else {
      resultsDiv.innerHTML = '<p style="color:#991b1b;">No result received. Please try again.</p>';
    }
  } catch (e) {
    resultsDiv.textContent = 'Error analyzing image.';
    document.getElementById('resultImages').style.display = 'none';
    document.getElementById('confidenceBar').style.display = 'none';
  }
});

// Voice input
voiceBtn.addEventListener("click", () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        alert("Speech Recognition is not supported.");
        return;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "en-IN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        console.log("Started listening...");
    };

    recognition.onaudiostart = () => {
        console.log("Audio started");
    };

    recognition.onspeechstart = () => {
        console.log("Speech detected");
    };

    recognition.onresult = (event) => {
        console.log("Recognized:", event.results[0][0].transcript);
        alert(event.results[0][0].transcript);
    };

    recognition.onerror = (event) => {
        console.error("Speech Error:", event.error);
        alert("Speech Error: " + event.error);
    };

    recognition.onend = () => {
        console.log("Recognition ended");
    };

    recognition.start();
});

// Stop button handler
stopBtn.addEventListener('click', ()=>{
  window.speechSynthesis.cancel();
  if(recognitionGlobal){
    try{ recognitionGlobal.abort(); }catch(e){}
  }
});

// Speak advice button
if (speakAdviceBtn) {
  speakAdviceBtn.addEventListener('click', () => {
    const text = resultsDiv.textContent || '';
    if (!text.trim() || text.includes('Awaiting input') || text.includes('Analyzing...') || text.includes('Processing query...')) {
      resultsDiv.textContent = 'No advice to speak yet. Please analyze a leaf or ask a question first.';
      return;
    }
    speakOut(text);
  });
}

// Text query submit (Enter key)
if (textQuery) {
  textQuery.addEventListener('keydown', async (e) => {
    if (e.key === 'Enter') {
      await handleTextQuery();
    }
  });
}

// Send button click
if (sendBtn) {
  sendBtn.addEventListener('click', async () => {
    await handleTextQuery();
  });
}

async function handleTextQuery() {
  const query = textQuery.value.trim();
  if (!query) return;
  resultsDiv.textContent = 'Processing query...';
  
  if (!PWA.isOnline()) {
    const result = await PWA.offlineFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, lang: userLang, lat: userLat, lon: userLon }),
    });
    if (result.queued) {
      resultsDiv.textContent = 'Offline: Message queued. Will send when connection returns.';
      return;
    }
  }

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, lang: userLang, lat: userLat, lon: userLon })
    });
    const data = await resp.json();
    resultsDiv.textContent = data.reply || 'No reply.';
  } catch (e) {
    resultsDiv.textContent = 'Error processing query.';
  }
}

// ---------------------------------------------------------------------------
// PWA Module
// ---------------------------------------------------------------------------
const PWA = (() => {
  let deferredPrompt = null;
  let isOnline = navigator.onLine;
  let swRegistration = null;
  let syncSupported = false;
  let notificationPermission = 'default';

  const DB_NAME = 'CHAI_OfflineDB';
  const DB_VERSION = 1;

  // IndexedDB helpers
  function openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
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
        if (!db.objectStoreNames.contains('cached-data')) {
          db.createObjectStore('cached-data', { keyPath: 'key' });
        }
      };
    });
  }

  async function addToPendingQueue(requestData) {
    const db = await openDB();
    const tx = db.transaction('pending-requests', 'readwrite');
    const store = tx.objectStore('pending-requests');
    await store.add({
      ...requestData,
      timestamp: Date.now(),
    });
  }

  async function getPendingRequests() {
    const db = await openDB();
    const tx = db.transaction('pending-requests', 'readonly');
    const store = tx.objectStore('pending-requests');
    return store.getAll();
  }

  async function removePendingRequest(id) {
    const db = await openDB();
    const tx = db.transaction('pending-requests', 'readwrite');
    const store = tx.objectStore('pending-requests');
    await store.delete(id);
  }

  async function cacheOfflineData(key, data) {
    const db = await openDB();
    const tx = db.transaction('cached-data', 'readwrite');
    const store = tx.objectStore('cached-data');
    await store.put({ key, data, timestamp: Date.now() });
  }

  async function getCachedData(key) {
    const db = await openDB();
    const tx = db.transaction('cached-data', 'readonly');
    const store = tx.objectStore('cached-data');
    return store.get(key);
  }

  async function syncPendingRequests() {
    const requests = await getPendingRequests();
    for (const req of requests) {
      try {
        const response = await fetch(req.url, {
          method: req.method,
          headers: req.headers,
          body: req.body,
        });
        if (response.ok || response.status === 400 || response.status === 401 || response.status === 403 || response.status === 422) {
          await removePendingRequest(req.id);
        }
      } catch (e) {
        console.error('[PWA] Sync failed for request', req.id, e);
      }
    }
  }

  // Service Worker Registration
  async function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) {
      console.log('[PWA] Service Worker not supported');
      return;
    }

    try {
      swRegistration = await navigator.serviceWorker.register('/sw.js');
      console.log('[PWA] Service Worker registered:', swRegistration.scope);

      syncSupported = 'SyncManager' in window;

      swRegistration.addEventListener('updatefound', () => {
        const newWorker = swRegistration.installing;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.onLine) {
            showUpdateAvailable();
          }
        });
      });

      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'SYNC_COMPLETE') {
          console.log('[PWA] Background sync complete for request', event.data.requestId);
        }
      });
    } catch (error) {
      console.error('[PWA] Service Worker registration failed:', error);
    }
  }

  // Install Prompt
  function setupInstallPrompt() {
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      updateInstallBanner();
    });

    window.addEventListener('appinstalled', () => {
      deferredPrompt = null;
      updateInstallBanner();
      console.log('[PWA] App installed');
    });
  }

  async function installApp() {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    console.log('[PWA] Install prompt outcome:', outcome);
    deferredPrompt = null;
    updateInstallBanner();
  }

  function updateInstallBanner() {
    const banner = document.getElementById('installBanner');
    if (!banner) return;
    if (deferredPrompt) {
      banner.classList.remove('hidden');
    } else {
      banner.classList.add('hidden');
    }
  }

  // Offline Detection
  function updateOnlineStatus() {
    isOnline = navigator.onLine;
    const indicator = document.getElementById('offlineIndicator');
    if (indicator) {
      if (isOnline) {
        indicator.classList.add('hidden');
      } else {
        indicator.classList.remove('hidden');
      }
    }
  }

  function setupOfflineDetection() {
    window.addEventListener('online', () => {
      updateOnlineStatus();
      syncPendingRequests();
    });
    window.addEventListener('offline', updateOnlineStatus);
    updateOnlineStatus();
  }

  // Notifications
  async function requestNotificationPermission() {
    if (!('Notification' in window)) {
      return false;
    }
    notificationPermission = await Notification.requestPermission();
    return notificationPermission === 'granted';
  }

  function showLocalNotification(title, options = {}) {
    if (notificationPermission !== 'granted') {
      return;
    }
    const notification = new Notification(title, {
      icon: '/static/icons/icon-192x192.png',
      badge: '/static/icons/icon-192x192.png',
      requireInteraction: true,
      ...options,
    });
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
    return notification;
  }

  function scheduleNotification(title, body, delayMinutes = 0, options = {}) {
    const delay = delayMinutes * 60 * 1000;
    setTimeout(() => {
      showLocalNotification(title, { body, ...options });
    }, delay);
  }

  function setupNotificationScheduler() {
    if (notificationPermission !== 'granted') {
      return;
    }

    scheduleNotification(
      'Weather Update',
      'Check the latest weather forecast for your area',
      5,
      { tag: 'weather-reminder' }
    );

    scheduleNotification(
      'CH AI Assistant',
      'Have you checked your crops today? Ask CH AI for advice',
      60,
      { tag: 'daily-check' }
    );
  }

  // Offline-aware fetch wrapper
  async function offlineFetch(url, options = {}) {
    if (!isOnline) {
      const cached = await getCachedData(url);
      if (cached) {
        return { data: cached.data, fromCache: true };
      }
      
      if (options.method === 'POST' && syncSupported) {
        await addToPendingQueue({
          url,
          method: options.method || 'GET',
          headers: options.headers || {},
          body: options.body,
        });
        if (swRegistration) {
          await swRegistration.sync.register('sync-pending-requests');
        }
        return { queued: true };
      }
      return { offline: true };
    }

    try {
      const response = await fetch(url, options);

let data = null;

if (response.ok) {
    data = await response.json();
    await cacheOfflineData(url, data);
}

return { data, fromCache: false };
    } catch (error) {
      const cached = await getCachedData(url);
      if (cached) {
        return { data: cached.data, fromCache: true };
      }
      throw error;
    }
  }

  // Show update available
  function showUpdateAvailable() {
    const banner = document.createElement('div');
    banner.className = 'fixed top-4 right-4 bg-yellow-600 text-white px-4 py-2 rounded shadow-lg z-50';
    banner.innerHTML = '<button id="reloadBtn" class="underline">Update available - reload</button>';
    document.body.appendChild(banner);
    document.getElementById('reloadBtn').addEventListener('click', () => {
      window.location.reload();
    });
  }

  // Initialize
  async function init() {
    await registerServiceWorker();
    setupInstallPrompt();
    setupOfflineDetection();
    
    if ('Notification' in window && Notification.permission === 'default') {
      await requestNotificationPermission();
    }

    if (notificationPermission === 'granted') {
      setupNotificationScheduler();
    }

    const installBtn = document.getElementById('installBtn');
    if (installBtn) {
      installBtn.addEventListener('click', installApp);
    }

    const dismissInstallBtn = document.getElementById('dismissInstallBtn');
    if (dismissInstallBtn) {
      dismissInstallBtn.addEventListener('click', () => {
        const banner = document.getElementById('installBanner');
        if (banner) banner.classList.add('hidden');
      });
    }
  }

  return {
    init,
    offlineFetch,
    isOnline: () => isOnline,
    requestNotificationPermission,
    showLocalNotification,
    syncPendingRequests,
  };
})();

// ---------------------------------------------------------------------------
// Training Monitor
// ---------------------------------------------------------------------------
const trainingModal = document.getElementById('trainingModal');
const showTrainingBtn = document.getElementById('showTrainingBtn');
const closeTrainingBtn = document.getElementById('closeTrainingBtn');
const startTrainingBtn = document.getElementById('startTrainingBtn');
const trainingStatusLabel = document.getElementById('trainingStatusLabel');
const trainingEpoch = document.getElementById('trainingEpoch');
const trainingTotalEpochs = document.getElementById('trainingTotalEpochs');
const trainingProgressPct = document.getElementById('trainingProgressPct');
const trainingProgressBar = document.getElementById('trainingProgressBar');
const trainingTrainAcc = document.getElementById('trainingTrainAcc');
const trainingValAcc = document.getElementById('trainingValAcc');
const trainingTrainLoss = document.getElementById('trainingTrainLoss');
const trainingValLoss = document.getElementById('trainingValLoss');
const trainingError = document.getElementById('trainingError');
const trainingCard = document.getElementById('trainingCard');
let trainingInterval = null;

function updateTrainingUI(data){
  if (!data) return
  const status = data.status || 'not_started'
  trainingStatusLabel.textContent = status === 'not_started' ? 'Not Started' : status === 'training' ? 'Training' : status === 'completed' ? 'Completed' : 'Failed'
  trainingEpoch.textContent = data.epoch || 0
  trainingTotalEpochs.textContent = data.total_epochs || 15
  const progress = data.total_epochs > 0 ? Math.min(100, Math.round(((data.epoch || 0) / data.total_epochs) * 100)) : 0
  trainingProgressPct.textContent = progress + '%'
  trainingProgressBar.style.width = progress + '%'
  const m = data.metrics || {}
  trainingTrainAcc.textContent = m.train_acc != null ? m.train_acc.toFixed(2) + '%' : '--'
  trainingValAcc.textContent = m.val_acc != null ? m.val_acc.toFixed(2) + '%' : '--'
  trainingTrainLoss.textContent = m.train_loss != null ? m.train_loss.toFixed(4) : '--'
  trainingValLoss.textContent = m.val_loss != null ? m.val_loss.toFixed(4) : '--'
  if (data.error) {
    trainingError.textContent = data.error
    trainingError.style.display = 'block'
  } else {
    trainingError.style.display = 'none'
  }
  if (status === 'training' && trainingCard) {
    trainingCard.style.display = 'block'
  }
}

async function fetchTrainingStatus(){
  try {
    const res = await fetch('/api/training/status')
    const data = await res.json()
    updateTrainingUI(data)
  } catch (e) {
    console.error('Training status fetch failed', e)
  }
}

function openTrainingModal(){
  if (trainingModal) {
    trainingModal.style.display = 'flex'
    fetchTrainingStatus()
    if (!trainingInterval) {
      trainingInterval = setInterval(fetchTrainingStatus, 2000)
    }
  }
}

function closeTrainingModal(){
  if (trainingModal) {
    trainingModal.style.display = 'none'
    if (trainingInterval) {
      clearInterval(trainingInterval)
      trainingInterval = null
    }
  }
}

async function startTraining(){
  try {
    const res = await fetch('/api/training/start', { method: 'POST' })
    const data = await res.json()
    if (!res.ok) {
      trainingError.textContent = data.error || 'Failed to start training'
      trainingError.style.display = 'block'
      return
    }
    trainingError.style.display = 'none'
    fetchTrainingStatus()
  } catch (e) {
    trainingError.textContent = 'Failed to start training'
    trainingError.style.display = 'block'
  }
}

if (showTrainingBtn) showTrainingBtn.addEventListener('click', openTrainingModal)
if (closeTrainingBtn) closeTrainingBtn.addEventListener('click', closeTrainingModal)
if (startTrainingBtn) startTrainingBtn.addEventListener('click', startTraining)
if (trainingModal) {
  trainingModal.addEventListener('click', (e) => {
    if (e.target === trainingModal) closeTrainingModal()
  })
}

// Initialize PWA on page load
document.addEventListener('DOMContentLoaded', () => {
  PWA.init();
  fetchTrainingStatus()
});



