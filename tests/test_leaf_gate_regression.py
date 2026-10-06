import io
import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_regression.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module

NON_LEAF_IMAGE = os.path.join(
    ROOT, 'artifacts', 'leaf_classifier', 'dataset', 'nonleaf', 'animal',
    '7-1-1000x680_10cde79598.jpg')

NON_LEAF_LOW_SCORE = os.path.join(
    ROOT, 'artifacts', 'leaf_classifier', 'dataset', 'nonleaf', 'animal',
    '230705_NVWG_PORTLAND_0310__53025872325__da72694b5b.jpg')

LEAF_IMAGE = None
_plantdoc = os.path.join(ROOT, 'data', 'PlantVillage', 'val')
if os.path.isdir(_plantdoc):
    for class_dir in sorted(os.listdir(_plantdoc)):
        full = os.path.join(_plantdoc, class_dir)
        if os.path.isdir(full):
            for name in sorted(os.listdir(full)):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    LEAF_IMAGE = os.path.join(full, name)
                    break
        if LEAF_IMAGE:
            break


def _post(client, path, **extra):
    with open(path, 'rb') as fh:
        payload = dict(image=(io.BytesIO(fh.read()), os.path.basename(path)), lang='en')
        payload.update(extra)
        return client.post('/api/analyze', data=payload,
                           content_type='multipart/form-data')


def test_non_leaf_never_reaches_disease_model():
    """The core regression: a non-leaf image must never invoke the disease classifier.

    We spy on app.predict_image (the disease-model inference function). When a
    non-leaf image is uploaded, that spy must record zero calls.
    """
    path = NON_LEAF_LOW_SCORE
    if not os.path.exists(path):
        pytest.skip('non-leaf fixture missing')

    client = app_module.app.test_client()
    with patch.object(app_module, 'predict_image', side_effect=AssertionError(
        'disease model must never run on a non-leaf image')) as spy:
        resp = _post(client, path, include_gradcam='true', source='upload')
    
    assert resp.status_code == 422
    payload = resp.get_json()
    assert payload.get('error_type') == 'NON_LEAF_IMAGE'
    assert not any(k in payload for k in ('class', 'confidence', 'remedy',
                                           'gradcam_overlay', 'top_k_classes'))


def test_green_dog_never_reaches_disease_model():
    """The green dog hard-negative (score 0.6101) must never reach the disease model.

    It falls in the UNCERTAIN band (above reject 0.1, below accept 0.62), so it is
    refused as LEAF_UNCERTAIN rather than NON_LEAF. Either way, the disease model
    must not run.
    """
    path = NON_LEAF_IMAGE
    if not os.path.exists(path):
        pytest.skip('green dog fixture missing')

    client = app_module.app.test_client()
    with patch.object(app_module, 'predict_image', side_effect=AssertionError(
        'disease model must never run on a non-leaf image')) as spy:
        resp = _post(client, path, include_gradcam='true', source='upload')
    
    assert resp.status_code == 422
    payload = resp.get_json()
    assert payload.get('error_type') in ('NON_LEAF_IMAGE', 'LEAF_UNCERTAIN')
    assert not any(k in payload for k in ('class', 'confidence', 'remedy',
                                           'gradcam_overlay', 'top_k_classes'))


def test_uncertain_image_never_reaches_disease_model():
    """An UNCERTAIN verdict must also block disease inference."""
    path = NON_LEAF_IMAGE
    if not os.path.exists(path):
        pytest.skip('non-leaf fixture missing')

    for validator in (app_module.LEAF_GATE,
                      getattr(app_module, 'LEAF_CLASSIFIER', None)):
        if validator is not None:
            validator.accept_threshold = 1.01
            validator.reject_threshold = 0.0

    try:
        client = app_module.app.test_client()
        with patch.object(app_module, 'predict_image', side_effect=AssertionError(
            'disease model must never run on an uncertain image')) as spy:
            resp = _post(client, path, include_gradcam='true', source='upload')
        
        assert resp.status_code == 422
        payload = resp.get_json()
        assert payload.get('error_type') == 'LEAF_UNCERTAIN'
    finally:
        for validator in (app_module.LEAF_GATE,
                          getattr(app_module, 'LEAF_CLASSIFIER', None)):
            if validator is not None:
                if validator is app_module.LEAF_GATE:
                    pass  # Gate thresholds restored from config on reload
                else:
                    validator.accept_threshold = 0.62
                    validator.reject_threshold = 0.1


def test_real_leaf_reaches_disease_model():
    """A genuine leaf must invoke the disease classifier."""
    path = LEAF_IMAGE
    if path is None:
        pytest.skip('PlantVillage validation split not present')

    client = app_module.app.test_client()
    with patch.object(app_module, 'predict_image', wraps=app_module.predict_image) as spy:
        resp = _post(client, path, include_gradcam='true', source='upload')
    
    assert spy.called, 'disease model must be called for a genuine leaf'
    assert resp.status_code == 200
    payload = resp.get_json()
    assert 'class' in payload
