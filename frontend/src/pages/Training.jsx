import React, { useEffect, useState, useRef } from 'react';

const STATUS_CONFIG = {
  not_started: { label: 'Not Started', color: 'bg-slate-100 text-slate-700 border-slate-200', dot: 'bg-slate-400' },
  training: { label: 'Training', color: 'bg-blue-50 text-blue-700 border-blue-200', dot: 'bg-blue-500 animate-pulse' },
  completed: { label: 'Completed', color: 'bg-emerald-50 text-emerald-700 border-emerald-200', dot: 'bg-emerald-500' },
  failed: { label: 'Failed', color: 'bg-red-50 text-red-700 border-red-200', dot: 'bg-red-500' },
};

const TRAINING_CONFIG = {
  model: 'MobileNetV2',
  classes: 38,
  imageSize: '224×224',
  batchSize: 32,
  learningRate: '1e-4',
  epochs: 15,
  optimizer: 'Adam',
  lossFunction: 'CrossEntropyLoss',
  validationSplit: '15%',
  scheduler: 'CosineAnnealingLR',
  earlyStopping: '3 epochs',
  randomSeed: 42,
};

function LineChart({ data, color, label, height = 240, width = 500 }) {
  if (!data || data.length < 2) {
    return (
      <div className="card p-5" style={{ height }}>
        <p className="text-sm font-medium text-slate-700 mb-3">{label}</p>
        <div className="flex items-center justify-center h-48 text-sm text-slate-400">
          Waiting for training data...
        </div>
      </div>
    );
  }

  const values = data.map((d) => d.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const points = data.map((d, i) => ({
    x: padding.left + (data.length === 1 ? 0 : (i / (data.length - 1)) * chartW),
    y: padding.top + chartH - ((d.value - min) / range) * chartH,
    value: d.value,
    epoch: d.epoch,
  }));

  const pathD = points.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');

  const yTicks = 5;
  const yLabels = Array.from({ length: yTicks }, (_, i) => min + (range * i) / (yTicks - 1));

  return (
    <div className="card p-5">
      <p className="text-sm font-medium text-slate-700 mb-4">{label}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#e2e8f0" strokeWidth="1" />
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#e2e8f0" strokeWidth="1" />

        {yLabels.map((val, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={padding.top + (chartH * i) / (yTicks - 1)}
              x2={width - padding.right}
              y2={padding.top + (chartH * i) / (yTicks - 1)}
              stroke="#f1f5f9"
              strokeWidth="1"
            />
            <text
              x={padding.left - 8}
              y={padding.top + (chartH * i) / (yTicks - 1) + 4}
              textAnchor="end"
              fontSize="10"
              fill="#64748b"
            >
              {val.toFixed(2)}
            </text>
          </g>
        ))}

        <path d={pathD} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r="4" fill={color} stroke="white" strokeWidth="2" />
        ))}

        {points.map((p, i) => (
          <text key={i} x={p.x} y={height - 8} textAnchor="middle" fontSize="10" fill="#64748b">
            {p.epoch}
          </text>
        ))}
      </svg>
    </div>
  );
}

function ConfigRow({ label, value }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-sm font-medium text-slate-900">{value}</span>
    </div>
  );
}

export default function Training() {
  const [status, setStatus] = useState('not_started');
  const [epoch, setEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(TRAINING_CONFIG.epochs);
  const [metrics, setMetrics] = useState({});
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const intervalRef = useRef(null);

  const loadStatus = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL || ''}/api/training/status`);
      const data = await res.json();
      setStatus(data.status || 'not_started');
      setEpoch(data.epoch || 0);
      setTotalEpochs(data.total_epochs || TRAINING_CONFIG.epochs);
      setMetrics(data.metrics || {});
      setError(data.error || '');

      if (data.metrics && data.metrics.val_acc != null) {
        setHistory((prev) => {
          const next = [...prev];
          if (next.length === 0 || next[next.length - 1].epoch !== data.epoch) {
            next.push({
              epoch: data.epoch,
              train_loss: data.metrics.train_loss,
              train_acc: data.metrics.train_acc,
              val_loss: data.metrics.val_loss,
              val_acc: data.metrics.val_acc,
            });
          }
          return next;
        });
      }
    } catch (e) {
      console.error('Failed to load training status', e);
    }
  };

  useEffect(() => {
    loadStatus();
    intervalRef.current = setInterval(loadStatus, 2000);
    return () => clearInterval(intervalRef.current);
  }, []);

  const startTraining = async () => {
    setError('');
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL || ''}/api/training/start`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Failed to start training');
        return;
      }
      setHistory([]);
      await loadStatus();
    } catch (err) {
      setError('Failed to start training');
    }
  };

  const progress = totalEpochs > 0 ? Math.min(100, Math.round((epoch / totalEpochs) * 100)) : 0;
  const statusConfig = STATUS_CONFIG[status] || STATUS_CONFIG.not_started;
  const accuracyData = history.map((h) => ({ epoch: h.epoch, value: h.val_acc != null ? h.val_acc : 0 }));
  const lossData = history.map((h) => ({ epoch: h.epoch, value: h.val_loss != null ? h.val_loss : 0 }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="md:col-span-2 lg:col-span-2 space-y-6">
          <div className="card p-6">
            <div className="flex items-start justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Model Training</h3>
                <p className="text-sm text-slate-500 mt-1">Train MobileNetV2 on PlantVillage dataset with real-time monitoring.</p>
              </div>
              <div className="flex items-center gap-3">
                <span className={`badge ${statusConfig.color}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${statusConfig.dot}`} />
                  {statusConfig.label}
                </span>
                <button
                  onClick={startTraining}
                  disabled={status === 'training'}
                  className="btn btn-primary"
                >
                  {status === 'training' ? 'Training...' : 'Start Training'}
                </button>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600">
                  Epoch <span className="font-semibold text-slate-900">{epoch}</span> / {totalEpochs}
                </span>
                <span className="font-semibold text-slate-900">{progress}%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-emerald-500 h-3 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
                {error}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            <MetricCard title="Training Accuracy" value={metrics.train_acc != null ? `${metrics.train_acc.toFixed(2)}%` : '--'} />
            <MetricCard title="Validation Accuracy" value={metrics.val_acc != null ? `${metrics.val_acc.toFixed(2)}%` : '--'} />
            <MetricCard title="Training Loss" value={metrics.train_loss != null ? metrics.train_loss.toFixed(4) : '--'} />
            <MetricCard title="Validation Loss" value={metrics.val_loss != null ? metrics.val_loss.toFixed(4) : '--'} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
            <div className="min-w-0">
              <LineChart data={accuracyData} color="#10b981" label="Accuracy vs Epoch" />
            </div>
            <div className="min-w-0">
              <LineChart data={lossData} color="#ef4444" label="Loss vs Epoch" />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-xl">🧠</span>
              <h4 className="text-base font-semibold text-slate-900">Model Overview</h4>
            </div>
            <div className="space-y-1">
              <ConfigRow label="Architecture" value={TRAINING_CONFIG.model} />
              <ConfigRow label="Classes" value={TRAINING_CONFIG.classes} />
              <ConfigRow label="Dataset" value="PlantVillage" />
              <ConfigRow label="Status" value={statusConfig.label} />
            </div>
          </div>

          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-xl">⚙️</span>
              <h4 className="text-base font-semibold text-slate-900">Training Configuration</h4>
            </div>
            <div className="space-y-1">
              <ConfigRow label="Learning Rate" value={TRAINING_CONFIG.learningRate} />
              <ConfigRow label="Epochs" value={TRAINING_CONFIG.epochs} />
              <ConfigRow label="Batch Size" value={TRAINING_CONFIG.batchSize} />
              <ConfigRow label="Optimizer" value={TRAINING_CONFIG.optimizer} />
              <ConfigRow label="Image Size" value={TRAINING_CONFIG.imageSize} />
              <ConfigRow label="Loss Function" value={TRAINING_CONFIG.lossFunction} />
              <ConfigRow label="Validation Split" value={TRAINING_CONFIG.validationSplit} />
              <ConfigRow label="Scheduler" value={TRAINING_CONFIG.scheduler} />
              <ConfigRow label="Early Stopping" value={TRAINING_CONFIG.earlyStopping} />
              <ConfigRow label="Random Seed" value={TRAINING_CONFIG.randomSeed} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value }) {
  return (
    <div className="card p-5">
      <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">{title}</p>
      <p className="text-xl font-semibold text-slate-900">{value}</p>
    </div>
  );
}
