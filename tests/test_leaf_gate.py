import io
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_leaf_gate.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module
import leaf_gate

PLANTVILLAGE_VAL = os.path.join(ROOT, 'data', 'PlantVillage', 'val')
NON_LEAF = os.path.join(ROOT, 'static', 'images')


def _first_leaf_image():
    """A real leaf photograph from the local PlantVillage split.

    The split is not tracked in git, so the gate tests skip rather than fail when the
    dataset is absent. A synthetic stand-in is not usable here: class coherence depends
    on what the real model predicts, which cannot be faked convincingly.
    """
    if not os.path.isdir(PLANTVILLAGE_VAL):
        return None
    for class_dir in sorted(os.listdir(PLANTVILLAGE_VAL)):
        full = os.path.join(PLANTVILLAGE_VAL, class_dir)
        if not os.path.isdir(full):
            continue
        for name in sorted(os.listdir(full)):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(full, name)
    return None


def _non_leaf_image():
    for name in sorted(os.listdir(NON_LEAF)):
        if name.startswith('Screenshot') and name.lower().endswith('.png'):
            return os.path.join(NON_LEAF, name)
    return None


def _post(client, path, **extra):
    with open(path, 'rb') as fh:
        payload = dict(image=(io.BytesIO(fh.read()), os.path.basename(path)), lang='en')
        payload.update(extra)
        return client.post('/api/analyze', data=payload,
                           content_type='multipart/form-data')


def test_gate_is_active_and_reports_its_configuration():
    assert app_module.LEAF_GATE is not None, 'the gate must load from artifacts/'
    assert app_module.LEAF_GATE_ENABLED, 'the gate must be on by default'

    client = app_module.app.test_client()
    resp = client.get('/api/leaf-gate/status')
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload['enabled'] is True
    assert payload['accept_threshold'] > payload['reject_threshold']
    assert payload['model_matches_audited'] is True
    assert payload['prototypes'] > 0


def test_non_leaf_photo_returns_no_diagnosis():
    """The whole point of the gate: no class, no confidence, no Grad-CAM, no remedy."""
    path = _non_leaf_image()
    if path is None:
        pytest.skip('no non-leaf fixture available')

    client = app_module.app.test_client()
    resp = _post(client, path, include_gradcam='true')
    payload = resp.get_json()

    assert resp.status_code == 422
    # The reason may be leaf suitability or image quality. Either is a refusal, and the
    # point of this test is that a refusal never carries a diagnosis.
    assert payload['error_type'] in (
        leaf_gate.ERROR_NON_LEAF,
        leaf_gate.ERROR_LEAF_UNCERTAIN,
        leaf_gate.ERROR_OVEREXPOSED,
        leaf_gate.ERROR_TOO_DARK,
        leaf_gate.ERROR_TOO_BLANK,
        leaf_gate.ERROR_TOO_BLURRY,
        leaf_gate.ERROR_TOO_SMALL,
        leaf_gate.ERROR_UNREADABLE,
    )
    assert payload['leaf_gate']['state'] in ('NON_LEAF', 'UNCERTAIN', 'REJECTED')
    assert payload['leaf_gate']['is_leaf'] is not True

    for forbidden in ('class', 'confidence', 'top_k_classes', 'remedy',
                      'gradcam_overlay', 'gradcam_heatmap', 'is_uncertain_calibrated'):
        assert forbidden not in payload, f'{forbidden} must not be emitted for a refused image'


def test_uncertain_image_carries_no_diagnosis_either(monkeypatch):
    """The UNCERTAIN state must be as diagnosis-free as an outright refusal.

    Which real photographs land in UNCERTAIN depends on the calibration data and on
    which validator is the authority, so the state is forced here by raising the accept
    threshold above any achievable score on whichever validator is actually running.
    The contract under test is the response shape, not where the line happens to sit.
    """
    # The authority is per-request and may be the heuristic gate or the classifier, so
    # widen both thresholds on every validator that is present. Raising accept alone is
    # not enough: a validator that is *confident* the image is not a leaf still returns
    # NON_LEAF, because that decision is taken at the reject threshold. Setting reject to
    # 0.0 makes "confidently not a leaf" unreachable, and accept to 1.01 makes "is a
    # leaf" unreachable, so every score lands in the UNCERTAIN band.
    for validator in (app_module.LEAF_GATE,
                      getattr(app_module, 'LEAF_CLASSIFIER', None)):
        if validator is not None:
            monkeypatch.setattr(validator, 'accept_threshold', 1.01)
            monkeypatch.setattr(validator, 'reject_threshold', 0.0)

    client = app_module.app.test_client()
    path = os.path.join(ROOT, 'static', 'images', '45908.jpg')
    if not os.path.exists(path):
        pytest.skip('fixture missing')

    resp = _post(client, path)
    payload = resp.get_json()

    assert resp.status_code == 422
    assert payload['error_type'] == leaf_gate.ERROR_LEAF_UNCERTAIN
    assert payload['leaf_gate']['state'] == 'UNCERTAIN'
    assert payload['leaf_gate']['is_leaf'] is None
    assert payload['is_uncertain'] is True
    for forbidden in ('class', 'confidence', 'top_k_classes', 'remedy',
                      'gradcam_overlay', 'gradcam_heatmap'):
        assert forbidden not in payload, f'{forbidden} must not be emitted when uncertain'


def test_real_leaf_photo_is_diagnosed():
    path = _first_leaf_image()
    if path is None:
        pytest.skip('PlantVillage validation split not present on this machine')

    client = app_module.app.test_client()
    resp = _post(client, path)
    payload = resp.get_json()

    assert resp.status_code == 200, payload.get('error')
    assert payload['class'] in app_module.idx_to_class.values()
    assert 0.0 <= payload['confidence'] <= 1.0
    assert payload['leaf_gate']['state'] == 'VALID_LEAF'
    assert payload['leaf_gate']['is_leaf'] is True


def test_refused_image_is_not_saved_to_history():
    """A refusal is not a prediction, so it must not pollute the history."""
    path = _non_leaf_image()
    if path is None:
        pytest.skip('no non-leaf fixture available')

    app_module.engine.dispose()
    app_module.Base.metadata.drop_all(bind=app_module.engine)
    app_module.Base.metadata.create_all(bind=app_module.engine)
    before = app_module.performance_metrics['images_refused_by_gate']

    client = app_module.app.test_client()
    resp = _post(client, path)
    assert resp.status_code == 422

    history = client.get('/api/predictions?limit=5').get_json()
    assert history['predictions'] == []
    assert app_module.performance_metrics['images_refused_by_gate'] == before + 1


def test_gate_threshold_config_never_refuses_a_measured_leaf():
    """The config-level guarantee: the reject line sits at or below the lowest leaf."""
    config_path = os.path.join(ROOT, 'artifacts', 'leaf_gate', 'leaf_gate_config.json')
    report_path = os.path.join(ROOT, 'experiments', 'leaf_gate', 'calibration_report.json')
    if not (os.path.exists(config_path) and os.path.exists(report_path)):
        pytest.skip('calibration artefacts not present')

    import json
    with open(config_path) as fh:
        config = json.load(fh)
    with open(report_path) as fh:
        report = json.load(fh)

    assert config['reject_threshold'] <= config['accept_threshold']
    assert config['model_sha256'] == app_module.AUDITED_MODEL_SHA256
    outcome = report['outcome']
    assert outcome['leaf_refused_by_score'] == 0
    assert outcome['leaf_refused_by_quality'] == 0
    # Every one of the 38 classes must be represented in the calibration sample.
    assert len(report['per_class_plantvillage']) == 38


def test_feature_transform_saturates_for_the_closest_leaf():
    """A leaf nearer to the prototypes than any stored example must score 1.0.

    This is the direction the transform has to get right; an inverted ramp silently
    scored the most leaf-like images at 0, which is how the original bug showed up.
    """
    gate = app_module.LEAF_GATE
    assert gate.feature_likeness(gate.distance_hi) == 1.0
    assert gate.feature_likeness(0.0) == 1.0
    assert gate.feature_likeness(gate.distance_lo) == 0.0
    assert gate.feature_likeness(gate.distance_lo * 10) == 0.0
    middle = gate.feature_likeness((gate.distance_hi + gate.distance_lo) / 2)
    assert 0.0 < middle < 1.0
    assert gate.feature_likeness(gate.distance_hi - 0.01) >= middle


def test_nonleaf_audit_batch():
    """Every image in the held-out audit non-leaf set must be refused by the classifier.

    The audit set (audit/test_set_manifest.json, non-leaf entries) is the safety net:
    it is distinct from the training/test split and contains photographs that must
    never reach the disease model. The dedicated leaf classifier is the primary
    gate when present, and this test verifies it admits zero of them.
    """
    manifest_path = os.path.join(ROOT, 'audit', 'test_set_manifest.json')
    if not os.path.exists(manifest_path):
        pytest.skip('audit test set manifest not present')

    import json
    with open(manifest_path) as fh:
        entries = json.load(fh)
    nonleaf = [e for e in entries if not e.get('expected_leaf')]

    if not nonleaf:
        pytest.skip('no non-leaf audit images')

    clf = getattr(app_module, 'LEAF_CLASSIFIER', None)
    if clf is None:
        pytest.skip('leaf classifier not loaded')

    audit_dir = os.path.join(ROOT, 'audit', 'test_set')
    refused = 0
    accepted = 0
    for entry in nonleaf:
        path = os.path.join(audit_dir, entry['file'])
        if not os.path.exists(path):
            continue
        with Image.open(path) as img:
            verdict = clf.classify(img.convert('RGB'))
        if verdict['state'] == 'LEAF':
            accepted += 1
        else:
            refused += 1

    assert accepted == 0, (
        f'{accepted} of {refused + accepted} audit non-leaf images were wrongly '
        f'accepted as leaves by the classifier')
    assert refused == len([e for e in nonleaf if os.path.exists(os.path.join(audit_dir, e['file']))])
