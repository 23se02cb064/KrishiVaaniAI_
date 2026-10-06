import React, { useRef, useState, useCallback } from 'react';
import CameraCapture from '../components/CameraCapture';

export default function Disease() {
  const fileRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [gradcamView, setGradcamView] = useState('blend');
  // Set when the backend refused the image, so the message can distinguish "this is not
  // a leaf" from "this leaf could not be diagnosed confidently" and from a transport
  // failure. The backend returns no class, confidence, Grad-CAM or remedy in all three.
  const [gateRefusal, setGateRefusal] = useState(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG, WebP).');
      return;
    }
    setError('');
    setResult(null);
    setGateRefusal(null);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const analyze = async (file, source = 'upload') => {
    if (!file) {
      setError('Please select a leaf image first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setGateRefusal(null);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('lang', 'en');
      formData.append('include_gradcam', 'true');
      formData.append('source', source);

      const res = await fetch('/api/analyze', { method: 'POST', body: formData });
      const data = await res.json();

      // The leaf gate refuses the image before classification with HTTP 422. That is a
      // deliberate "no diagnosis", not a server fault, so it gets its own presentation
      // instead of being reported as an analysis error.
      if (!res.ok) {
        if (data && data.error_type) {
          setGateRefusal({ message: data.error, errorType: data.error_type });
          return;
        }
        throw new Error((data && data.error) || 'Analysis failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCameraCapture = useCallback((file) => {
    setShowCamera(false);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
    analyze(file);
  }, [analyze]);

  const reset = () => {
    setResult(null);
    setPreview(null);
    setSelectedFile(null);
    setError('');
    setShowCamera(false);
    if (fileRef.current) fileRef.current.value = '';
  };

  const parseClass = (className) => {
    if (!className) return { crop: 'Unknown', disease: 'Unknown' };
    const parts = className.split('___');
    const crop = parts[0] || 'Unknown';
    const disease = parts[1] ? parts[1].replace(/_/g, ' ') : className;
    return { crop, disease };
  };

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : 0;
  const { crop, disease } = parseClass(result?.class);
  const isHealthy = !!result && !result.is_uncertain && disease.toLowerCase() === 'healthy';
  const isLeafUncertain = gateRefusal?.errorType === 'LEAF_UNCERTAIN';
  const isQualityRefusal = !!gateRefusal && /^IMAGE_/.test(gateRefusal.errorType || '');

  return (
    <div className="space-y-6">
      <div className="card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-xl">🔬</div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">AI Disease Diagnosis</h3>
            <p className="text-sm text-slate-500">Upload a leaf image or capture using camera to detect diseases and visualize affected regions.</p>
          </div>
        </div>

        {!showCamera ? (
          <>
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-colors cursor-pointer ${
                dragOver
                  ? 'border-emerald-400 bg-emerald-50'
                  : 'border-slate-300 hover:border-emerald-400 hover:bg-slate-50'
              }`}
              onClick={() => fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />
              <div className="text-4xl mb-3">📷</div>
              <p className="text-sm font-medium text-slate-700">
                {preview ? 'Click or drag to change image' : 'Click to upload or drag & drop a leaf image'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Supports JPG, PNG, WebP up to 16MB</p>
            </div>

            <div className="mt-4 flex gap-3">
              <button
                onClick={() => setShowCamera(true)}
                className="btn btn-outline flex-1"
              >
                <span className="mr-2">📹</span> Capture with Camera
              </button>
            </div>

            {preview && !result && selectedFile && (
              <div className="mt-4 flex gap-3">
                <button
                  onClick={() => analyze(selectedFile)}
                  disabled={loading}
                  className="btn btn-primary flex-1"
                >
                  {loading ? 'Analyzing...' : 'Analyze'}
                </button>
              </div>
            )}

            {(result || preview) && (
              <div className="mt-4 flex gap-3">
                <button onClick={reset} className="btn btn-secondary">
                  Analyze Another
                </button>
              </div>
            )}
          </>
        ) : (
          <CameraCapture onCapture={handleCameraCapture} onClose={() => setShowCamera(false)} />
        )}

        {gateRefusal && (
          <div
            className={`mt-4 p-4 border rounded-xl text-sm ${
              isLeafUncertain
                ? 'bg-amber-50 border-amber-200 text-amber-800'
                : 'bg-red-50 border-red-200 text-red-700'
            }`}
          >
            <p className="font-semibold mb-1">
              {isLeafUncertain
                ? 'Could not confidently identify a leaf'
                : isQualityRefusal
                  ? 'Image cannot be analysed'
                  : 'This does not look like a plant leaf image'}
            </p>
            <p className="mb-2">{gateRefusal.message}</p>
            <p className="font-medium">For a diagnosis, try:</p>
            <ul className="list-disc list-inside mt-1 space-y-0.5">
              <li>A close-up of a single leaf, filling most of the frame</li>
              <li>Even lighting, without harsh glare or deep shadow</li>
              <li>The whole leaf and any spots or lesions in focus</li>
            </ul>
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
            {error}
          </div>
        )}
      </div>

        {result && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-6">
                <div className="card p-6">
                  <h4 className="text-base font-semibold text-slate-900 mb-4">Diagnosis Result</h4>
                  {isHealthy && (
                    <div className="mb-4 p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
                      <p className="font-semibold text-emerald-800 text-sm">
                        No disease detected. The uploaded leaf appears healthy.
                      </p>
                      <p className="text-xs text-emerald-700 mt-1">
                        Confidence {confidencePercent}%. Keep monitoring the plant, and
                        re-check if symptoms appear, because a healthy-looking leaf does
                        not rule out early infection.
                      </p>
                    </div>
                  )}
                  {result.is_uncertain && (
                    <div className="mb-4 p-4 bg-amber-50 border border-amber-200 rounded-xl">
                      <div className="flex items-start gap-3">
                        <span className="text-2xl flex-shrink-0">⚠</span>
                        <div>
                          <p className="font-semibold text-amber-800 text-sm mb-1">Uncertain Prediction</p>
                          <p className="text-xs text-amber-700">
                            The AI model&apos;s confidence ({confidencePercent}%) is below the threshold
                            for this prediction. Consider capturing a clearer image or consulting a local expert.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                      <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Crop</p>
                      <p className="text-lg font-semibold text-slate-900">{crop}</p>
                    </div>
                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                      <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Disease</p>
                      <p className="text-lg font-semibold text-slate-900">{disease}</p>
                    </div>
                    <div className="sm:col-span-2 p-4 bg-slate-50 rounded-xl border border-slate-200">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-xs text-slate-500 uppercase tracking-wide">Confidence</p>
                        <span className={`text-lg font-semibold ${result.is_uncertain ? 'text-amber-600' : 'text-emerald-600'}`}>{confidencePercent}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                        <div
                          className={`h-3 rounded-full transition-all duration-700 ease-out ${result.is_uncertain ? 'bg-amber-500' : 'bg-emerald-500'}`}
                          style={{ width: `${confidencePercent}%` }}
                        />
                      </div>
                      {result.confidence_threshold && (
                        <p className="text-xs text-slate-400 mt-1">
                          Threshold: {(result.confidence_threshold * 100).toFixed(0)}%
                        </p>
                      )}
                    </div>
                    {result.top_k_classes && result.top_k_classes.length > 1 && (
                      <div className="sm:col-span-2 p-4 bg-slate-50 rounded-xl border border-slate-200">
                        <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Top Predictions</p>
                        <div className="space-y-1">
                          {result.top_k_classes.slice(0, 3).map((pred, i) => {
                            const predParts = pred.class.split('___');
                            const predName = predParts[1] ? predParts[1].replace(/_/g, ' ') : pred.class;
                            return (
                              <div key={i} className="flex justify-between text-sm">
                                <span className="text-slate-700">
                                  #{i + 1} {predName}
                                </span>
                                <span className="text-slate-500 font-medium">
                                  {(pred.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

              <div className="card p-6">
                <h4 className="text-base font-semibold text-slate-900 mb-4">Visual Analysis (Grad-CAM)</h4>
                <div className="flex gap-2 mb-4 p-1 bg-slate-100 rounded-lg text-xs">
                  <button
                    onClick={() => setGradcamView('side-by-side')}
                    className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                      gradcamView === 'side-by-side'
                        ? 'bg-emerald-500 text-white'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Side-by-Side
                  </button>
                  <button
                    onClick={() => setGradcamView('blend')}
                    className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                      gradcamView === 'blend'
                        ? 'bg-emerald-500 text-white'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Overlay
                  </button>
                  <button
                    onClick={() => setGradcamView('heatmap')}
                    className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                      gradcamView === 'heatmap'
                        ? 'bg-emerald-500 text-white'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Heatmap
                  </button>
                </div>
                {gradcamView === 'side-by-side' ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-slate-500 mb-2 font-medium">Original Leaf</p>
                      <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
                        <img
                          src={preview}
                          alt="Uploaded leaf"
                          className="w-full h-48 object-cover"
                        />
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500 mb-2 font-medium">Affected Region</p>
                      <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
                        {result.gradcam_overlay ? (
                          <img
                            src={`data:image/png;base64,${result.gradcam_overlay}`}
                            alt="AI attention heatmap"
                            className="w-full h-48 object-cover"
                          />
                        ) : (
                          <div className="w-full h-48 flex items-center justify-center text-sm text-slate-400">
                            Heatmap not available
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
                    {gradcamView === 'blend' ? (
                      result.gradcam_overlay ? (
                        <img
                          src={`data:image/png;base64,${result.gradcam_overlay}`}
                          alt="Leaf with AI attention overlay"
                          className="w-full h-64 object-cover"
                        />
                      ) : (
                        <img
                          src={preview}
                          alt="Uploaded leaf"
                          className="w-full h-64 object-cover"
                        />
                      )
                    ) : (
                      result.gradcam_heatmap ? (
                        <img
                          src={`data:image/png;base64,${result.gradcam_heatmap}`}
                          alt="AI attention heatmap"
                          className="w-full h-64 object-cover"
                        />
                      ) : (
                        <div className="w-full h-64 flex items-center justify-center text-sm text-slate-400">
                          Heatmap not available
                        </div>
                      )
                    )}
                    <p className="absolute bottom-2 left-2 text-xs text-white bg-black/50 px-2 py-1 rounded">
                      {gradcamView === 'blend' ? 'Original + AI attention (blend)' : 'AI attention heatmap'}
                    </p>
                  </div>
                )}
                <p className="text-xs text-slate-500 mt-3 italic">
                  Highlighted areas indicate regions that contributed most to the AI model&apos;s prediction.
                </p>
              </div>
            </div>

            <div className="space-y-6">
              <div className="card p-6">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">🩺</span>
                  <h4 className="text-base font-semibold text-slate-900">Disease Information</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Detected Condition</p>
                    <p className="text-sm font-medium text-slate-900">{disease}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Affected Crop</p>
                    <p className="text-sm font-medium text-slate-900">{crop}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">AI Confidence</p>
                    <p className="text-sm font-medium text-emerald-600">{confidencePercent}%</p>
                  </div>
                </div>
              </div>

              <div className="card p-6">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">💊</span>
                  <h4 className="text-base font-semibold text-slate-900">Recommendation</h4>
                </div>
                <p className="text-sm text-slate-600 leading-relaxed">{result.remedy}</p>
              </div>

              {result.gradcam_error && (
                <div className="card p-4 border-amber-200 bg-amber-50">
                  <p className="text-xs text-amber-700">{result.gradcam_error}</p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}