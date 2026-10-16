import io
import json
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
from leaf_classifier import get_classifier, STATE_LEAF, STATE_NON_LEAF, STATE_UNCERTAIN

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


def test_classifier_class_mapping_is_correct():
    """Index 0 must mean nonleaf and index 1 must mean leaf.

    The checkpoint stores labels ['nonleaf', 'leaf'] and probability()
    reads probs[0, 1] as the leaf score. If this mapping were inverted,
    non-leaf images would be accepted as leaves.
    """
    clf = get_classifier()
    if clf is None:
        pytest.skip('leaf classifier not loaded')
    assert clf.labels == ['nonleaf', 'leaf'], f'unexpected labels: {clf.labels}'
    assert clf.labels[0] == 'nonleaf'
    assert clf.labels[1] == 'leaf'


def test_classifier_fail_closed_on_inference_error(monkeypatch):
    """If the classifier throws during inference, it must return UNCERTAIN,
    never LEAF. A fail-open error would let non-leaf images reach the
    disease model.
    """
    clf = get_classifier()
    if clf is None:
        pytest.skip('leaf classifier not loaded')

    original_probability = clf.probability

    def _failing_probability(image):
        raise RuntimeError('simulated inference failure')

    monkeypatch.setattr(clf, 'probability', _failing_probability)

    try:
        from PIL import Image
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        verdict = clf.classify(img)
    finally:
        monkeypatch.setattr(clf, 'probability', original_probability)

    assert verdict['state'] == STATE_UNCERTAIN, \
        f'classifier must fail closed to UNCERTAIN, got {verdict["state"]}'
    assert verdict['is_leaf'] is None


def test_classifier_rejects_all_audit_nonleaf_images():
    """Every non-leaf image in the held-out audit test set must be rejected.

    This is the safety net: zero non-leaf false accepts. The production
    incident involved a non-leaf image type not present in this audit set,
    so this test guards against regression on the images we *have* measured.
    """
    clf = get_classifier()
    if clf is None:
        pytest.skip('leaf classifier not loaded')

    manifest_path = os.path.join(ROOT, 'audit', 'test_set_manifest.json')
    if not os.path.exists(manifest_path):
        pytest.skip('audit test set manifest not present')

    with open(manifest_path) as fh:
        entries = json.load(fh)

    audit_dir = os.path.join(ROOT, 'audit', 'test_set')
    nonleaf = [e for e in entries if not e.get('expected_leaf')]
    if not nonleaf:
        pytest.skip('no non-leaf audit images')

    accepted = 0
    refused = 0
    for entry in nonleaf:
        path = os.path.join(audit_dir, entry['file'])
        if not os.path.exists(path):
            continue
        with Image.open(path) as img:
            verdict = clf.classify(img.convert('RGB'))
        if verdict['state'] == STATE_LEAF:
            accepted += 1
        else:
            refused += 1

    assert accepted == 0, (
        f'{accepted} of {refused + accepted} audit non-leaf images were wrongly '
        f'accepted as leaves by the classifier')
    assert refused == len([e for e in nonleaf if os.path.exists(os.path.join(audit_dir, e['file']))])


def test_classifier_accepts_most_audit_leaf_images():
    """Most genuine leaf images in the audit set must be accepted.

    Two red/yellow leaves are known to score below threshold (they lack green),
    so not all 18 will pass. This test guards against regression where
    previously-accepted leaves start being rejected.
    """
    clf = get_classifier()
    if clf is None:
        pytest.skip('leaf classifier not loaded')

    manifest_path = os.path.join(ROOT, 'audit', 'test_set_manifest.json')
    if not os.path.exists(manifest_path):
        pytest.skip('audit test set manifest not present')

    with open(manifest_path) as fh:
        entries = json.load(fh)

    audit_dir = os.path.join(ROOT, 'audit', 'test_set')
    leaf_entries = [e for e in entries if e.get('expected_leaf')]
    if not leaf_entries:
        pytest.skip('no leaf audit images')

    accepted = 0
    refused = 0
    for entry in leaf_entries:
        path = os.path.join(audit_dir, entry['file'])
        if not os.path.exists(path):
            continue
        with Image.open(path) as img:
            verdict = clf.classify(img.convert('RGB'))
        if verdict['state'] == STATE_LEAF:
            accepted += 1
        else:
            refused += 1

    total = accepted + refused
    assert total > 0, 'no audit leaf images were readable'
    assert accepted >= total * 0.75, (
        f'only {accepted}/{total} audit leaf images accepted (< 75%)')
