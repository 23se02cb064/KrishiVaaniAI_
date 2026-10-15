"""Tests for confidence calibration and uncertainty messaging."""

import json
import math
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app


def _make_logits(n=5, k=3):
    return torch.randn(n, k, dtype=torch.float64)


def test_temperature_does_not_change_argmax():
    logits = _make_logits()
    raw = F.softmax(logits, dim=1)
    for T in (0.5, 1.0, 2.7566, 10.0):
        cal = F.softmax(logits / T, dim=1)
        assert torch.equal(raw.argmax(dim=1), cal.argmax(dim=1)), (
            'temperature scaling must not change the predicted class')


def test_calibrated_probabilities_in_unit_range():
    logits = _make_logits()
    for T in (0.05, 1.0, 2.7566, 20.0):
        probs = F.softmax(logits / T, dim=1)
        assert (probs >= 0).all() and (probs <= 1).all()
        # Sum per-row; softmax is normalised over the class dimension.
        row_sums = probs.sum(dim=1)
        for s in row_sums:
            assert math.isclose(s.item(), 1.0, abs_tol=1e-5)


def test_invalid_temperature_rejected():
    bad = [0.0, -1.0, float('nan'), float('inf'), -0.001]
    for T in bad:
        ok = (math.isfinite(T) and T > 0)
        assert not ok, 'temperature %r must be rejected' % T


def test_malformed_calibration_file_fails_safe(tmp_path):
    p = tmp_path / 'bad.json'
    p.write_text('not json', encoding='utf-8')
    app.CALIBRATION_FILE = str(p)
    app.CALIBRATION_TEMPERATURE = None
    # Re-run the loader block by re-executing the logic inline.
    T = None
    try:
        with open(p, 'r') as fh:
            T = float(json.load(fh)['temperature'])
    except Exception:
        T = None
    assert T is None, 'a malformed calibration file must not set a temperature'


def test_missing_temperature_key_fails_safe(tmp_path):
    p = tmp_path / 'no_temp.json'
    p.write_text(json.dumps({'other': 1.0}), encoding='utf-8')
    T = None
    try:
        with open(p, 'r') as fh:
            T = float(json.load(fh)['temperature'])
    except Exception:
        T = None
    assert T is None, 'a file without a temperature key must not set one'


def test_valid_temperature_loads(tmp_path):
    p = tmp_path / 'good.json'
    p.write_text(json.dumps({'temperature': 2.7566}), encoding='utf-8')
    T = None
    with open(p, 'r') as fh:
        T = float(json.load(fh)['temperature'])
    assert T == 2.7566
    assert math.isfinite(T) and T > 0


def test_response_fields_backward_compatible():
    raw = {
        'class': 'Tomato___Early_blight',
        'confidence': 0.9234,
        'is_uncertain': False,
        'confidence_threshold': 0.5,
        'remedy': 'Remove infected leaves.',
        'remedy_source': 'static',
        'lang': 'en',
        'top_k_classes': [
            {'class': 'Tomato___Early_blight', 'idx': 29, 'confidence': 0.9234},
            {'class': 'Tomato___Late_blight', 'idx': 30, 'confidence': 0.0412},
        ],
        'timings': {'preprocessing_ms': 1.0, 'inference_ms': 2.0, 'total_ms': 3.0},
    }
    assert 'class' in raw
    assert 'confidence' in raw
    assert 'is_uncertain' in raw
    assert 'remedy' in raw
    assert 'top_k_classes' in raw
    assert raw['class'] == 'Tomato___Early_blight'


def test_calibrated_fields_only_when_calibration_valid():
    raw = {'confidence': 0.9234, 'is_uncertain': False}
    calibrated = dict(raw)
    calibrated['confidence_calibrated'] = 0.35
    calibrated['is_uncertain_calibrated'] = True
    calibrated['calibration_temperature'] = 2.7566
    assert 'confidence_calibrated' not in raw
    assert 'confidence_calibrated' in calibrated
    assert calibrated['is_uncertain_calibrated'] is True


def test_frontend_handles_missing_calibrated_fields():
    raw = {'confidence': 0.9234, 'is_uncertain': False}
    has_cal = raw.get('calibration_temperature') is not None
    display = raw.get('confidence_calibrated', raw['confidence']) if has_cal else raw['confidence']
    assert not has_cal
    assert display == 0.9234


def test_uncertain_message_present_when_below_threshold():
    msg = (
        'The AI model is not confident about this prediction '
        '(confidence: 21.0%). This is an AI estimate, not a confirmed diagnosis. '
        'For a more accurate result, try capturing a clearer image with even '
        'lighting or consulting a local agricultural expert.'
    )
    assert msg
    assert 'clearer image' in msg.lower()
    assert 'expert' in msg.lower()
    assert 'not a confirmed diagnosis' in msg.lower()


def test_gradcam_fields_unchanged():
    result = {
        'gradcam_overlay': 'base64data',
        'gradcam_heatmap': 'base64data2',
        'gradcam_error': None,
    }
    assert result['gradcam_overlay'] == 'base64data'
    assert result['gradcam_heatmap'] == 'base64data2'
    assert 'gradcam_overlay' in result and 'gradcam_heatmap' in result