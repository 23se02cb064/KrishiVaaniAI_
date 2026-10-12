import React, { useRef, useState, useEffect, useCallback } from 'react';

// Preferred constraints first, then progressively looser ones. `{ video: true }`
// is the last resort so a device that rejects facingMode or a specific size can
// still produce a live preview.
const ENV_CONSTRAINTS = {
  video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
};
const SIZED_CONSTRAINTS = {
  video: { width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
};
const MINIMAL_CONSTRAINTS = { video: true, audio: false };

const STOP_TO_RESTART_DELAY_MS = 80;

function describeCameraError(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'Camera permission denied. Allow camera access for this site in your browser settings, then try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera was found on this device.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The camera is already in use by another application. Close the other app or browser tab using it, then try again.';
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return 'This camera does not support the requested settings. Retake will try again with default settings.';
    case 'AbortError':
      return 'The camera stream was interrupted before it started. Tap Retake to try again.';
    case 'UnsupportedError':
      return 'Camera capture is not supported in this browser.';
    default:
      return `Failed to start the camera: ${(err && err.message) || 'unknown error'}`;
  }
}

// Waits for the element to have media before calling play(). Calling play()
// straight after assigning srcObject is what produces AbortError on some
// browsers, because the element has no metadata yet and autoplay is then
// interrupted when the source is replaced.
function waitForMetadata(video, isStale) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
    };
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error('Video metadata load failed'));
    };

    if (isStale()) {
      resolve();
      return;
    }
    if (video.readyState >= 1) {
      resolve();
      return;
    }
    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('error', onError);
  });
}

async function playVideo(video, stream, isStale) {
  try {
    await video.play();
    return;
  } catch (playErr) {
    if (isStale()) return;
    if (!playErr || playErr.name !== 'AbortError') throw playErr;
  }

  // AbortError means the play request was cancelled, typically because the
  // element was re-attached a moment earlier. One retry after a short pause is
  // enough; a second failure is surfaced to the caller.
  await new Promise(r => setTimeout(r, 120));
  if (isStale() || video.srcObject !== stream) return;
  await video.play();
}

export default function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const isMountedRef = useRef(true);
  const startingRef = useRef(false);
  const generationRef = useRef(0);
  const playAttemptedRef = useRef(false);

  // The MediaStream itself lives in streamRef only: keeping it in state would
  // re-render on every track change and tempt an effect to depend on it.
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [showGuide, setShowGuide] = useState(true);

  const qualityTips = [
    'Place the leaf flat against a neutral background',
    'Use even lighting - avoid harsh shadows or direct flash',
    'Fill the frame with the leaf, showing the entire leaf surface',
    'Keep the camera steady and hold for 1-2 seconds after capturing',
    'Ensure both the upper and lower surfaces of the leaf are visible if possible',
    'Avoid glare or reflections on the leaf surface',
  ];

  const stopCamera = useCallback(() => {
    // Bump the generation first: any start still awaiting getUserMedia or
    // play() sees itself as stale and releases whatever it acquired.
    generationRef.current += 1;
    startingRef.current = false;

    const currentStream = streamRef.current;
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
    playAttemptedRef.current = false;

    if (isMountedRef.current) {
      setCameraActive(false);
      setStarting(false);
    }
  }, []);

  const startCamera = useCallback(async () => {
    if (!isMountedRef.current) return;
    if (startingRef.current) return;   // never run two getUserMedia calls at once
    if (streamRef.current) return;     // already streaming

    startingRef.current = true;
    const generation = ++generationRef.current;
    const isStale = () => !isMountedRef.current || generationRef.current !== generation;
    const releaseStream = (s) => s && s.getTracks().forEach(t => t.stop());

    setStarting(true);
    setError('');
    playAttemptedRef.current = true;

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const unsupported = new Error('Camera API not supported in this browser');
        unsupported.name = 'UnsupportedError';
        throw unsupported;
      }

      let mediaStream;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia(ENV_CONSTRAINTS);
      } catch (envErr) {
        if (isStale()) return;
        console.warn('Rear camera unavailable, retrying with looser video constraints:', envErr.name);
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia(SIZED_CONSTRAINTS);
        } catch (sizeErr) {
          if (isStale()) return;
          console.warn('Sized constraints unavailable, retrying with { video: true }:', sizeErr.name);
          mediaStream = await navigator.mediaDevices.getUserMedia(MINIMAL_CONSTRAINTS);
        }
      }

      if (isStale()) {
        releaseStream(mediaStream);
        return;
      }

      const video = videoRef.current;
      if (!video) {
        releaseStream(mediaStream);
        return;
      }

      streamRef.current = mediaStream;

      video.srcObject = mediaStream;
      await waitForMetadata(video, isStale);
      if (isStale()) {
        releaseStream(mediaStream);
        return;
      }

      await playVideo(video, mediaStream, isStale);
      if (isStale()) {
        releaseStream(mediaStream);
        return;
      }

      playAttemptedRef.current = true;
      setCapturedImage(null);
      setCameraActive(true);
    } catch (err) {
      console.error('Camera start failed:', err);
      playAttemptedRef.current = false;
      const failed = streamRef.current;
      releaseStream(failed);
      streamRef.current = null;
      const video = videoRef.current;
      if (video) video.srcObject = null;

      if (isMountedRef.current) {
        setCameraActive(false);
        setError(describeCameraError(err));
      }
    } finally {
      startingRef.current = false;
      if (isMountedRef.current && generationRef.current === generation) {
        setStarting(false);
      }
    }
  }, []);

  const capturePhoto = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !streamRef.current) return;

    if (!video.videoWidth || !video.videoHeight) {
      setError('The camera is not ready yet. Wait for the live preview, then capture again.');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    setError('');
    setCapturedImage(canvas.toDataURL('image/jpeg', 0.9));
  }, []);

  const retakePhoto = useCallback(async () => {
    // Clear the captured frame first so the UI leaves the review state even if
    // the new stream takes a moment to arrive.
    setCapturedImage(null);

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      canvas.width = 0;
      canvas.height = 0;
    }

    stopCamera();

    // Give the browser a tick to release the previous capture device; asking
    // for a new stream in the same task can yield NotReadableError on Windows
    // and Safari.
    await new Promise(r => setTimeout(r, STOP_TO_RESTART_DELAY_MS));
    if (!isMountedRef.current) return;

    startCamera();
  }, [stopCamera, startCamera]);

  const analyzePhoto = useCallback(async () => {
    if (!capturedImage) return;

    try {
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
      onCapture(file);
    } catch (err) {
      console.error('Failed to process captured image:', err);
      setError('Failed to process captured image.');
    }
  }, [capturedImage, onCapture]);

  useEffect(() => {
    isMountedRef.current = true;
    startCamera();

    return () => {
      isMountedRef.current = false;
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  const handleClose = useCallback(() => {
    stopCamera();
    onClose();
  }, [stopCamera, onClose]);

  return (
    <div className="space-y-4">
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="relative bg-black rounded-xl overflow-hidden">
        {/* The <video> element stays mounted for the whole session. React would
            otherwise unmount it while a capture is on screen and hand a brand
            new element with a null srcObject back on Retake, which shows a
            black frame with no live video. */}
        <video
          ref={videoRef}
          className={capturedImage ? 'hidden' : 'w-full h-auto max-h-[500px] object-cover'}
          playsInline
          muted
          autoPlay
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {capturedImage && (
          <img
            src={capturedImage}
            alt="Captured leaf"
            className="w-full h-auto max-h-[500px] object-contain"
          />
        )}

        {!capturedImage && showGuide && !starting && cameraActive && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-3 left-3 right-3 bg-black/60 text-white p-2.5 rounded-lg text-xs">
              <div className="flex items-center gap-2 mb-1.5">
                <span>📸</span>
                <strong>Capture Tips:</strong>
              </div>
              <ul className="space-y-0.5 pl-4 list-disc list-inside">
                {qualityTips.map((tip, i) => (
                  <li key={i}>{tip}</li>
                ))}
              </ul>
            </div>
            <div className="absolute bottom-16 left-0 right-0 flex justify-center">
              <div className="w-64 h-48 border-2 border-dashed border-white/50 rounded-lg" />
            </div>
          </div>
        )}

        {capturedImage ? (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent">
            <div className="flex gap-3 justify-center">
              <button
                onClick={retakePhoto}
                className="btn btn-secondary px-6"
              >
                Retake
              </button>
              <button
                onClick={analyzePhoto}
                disabled={starting}
                className="btn btn-primary px-6"
              >
                {starting ? 'Analyzing...' : 'Analyze Leaf'}
              </button>
            </div>
          </div>
        ) : starting ? (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="text-white text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-white/30 border-t-white mx-auto mb-3" />
              <p>Starting camera...</p>
            </div>
          </div>
        ) : cameraActive ? (
          <div className="absolute bottom-4 left-4 right-4 flex justify-center">
            <button
              onClick={() => setShowGuide(!showGuide)}
              className="absolute top-2 right-2 p-1.5 bg-black/40 text-white rounded-full hover:bg-black/60 transition-colors z-10"
              title={showGuide ? 'Hide capture guide' : 'Show capture guide'}
            >
              {showGuide ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="16" x2="12" y2="12" />
                  <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
              )}
            </button>
            <button
              onClick={capturePhoto}
              disabled={starting}
              className="btn btn-primary px-8 py-3 text-lg shadow-lg"
            >
              Capture Leaf
            </button>
          </div>
        ) : null}

        <button
          onClick={handleClose}
          className="absolute top-3 right-3 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
          aria-label="Close camera"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <p className="text-xs text-slate-500 text-center">
        {capturedImage ? 'Review the captured image above. Retake or analyze.' : 'Point camera at a leaf and tap Capture Leaf. Capture tips are shown in the overlay.'}
      </p>
    </div>
  );
}
