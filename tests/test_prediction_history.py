import io
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_history.db')
os.environ.setdefault('FLASK_TESTING', '1')

for path in ['test_history.db', 'test_history.db-journal']:
    if os.path.exists(os.path.join(ROOT, path)):
        os.remove(os.path.join(ROOT, path))

import app as app_module


def _use_ungated_analysis(monkeypatch):
    """These tests cover history persistence and export, not leaf suitability.

    The stand-in image is a flat green square, which the leaf gate correctly refuses as
    a non-leaf photograph, so the gate is switched off here to keep the test focused on
    the behaviour it is actually about. The gate has its own tests, in
    tests/test_leaf_gate.py.
    """
    monkeypatch.setattr(app_module, 'LEAF_VALIDATION_BYPASS', True)


def test_prediction_history_is_persisted_and_listed(monkeypatch):
    _use_ungated_analysis(monkeypatch)
    app_module.engine.dispose()
    app_module.Base.metadata.drop_all(bind=app_module.engine)
    app_module.Base.metadata.create_all(bind=app_module.engine)

    client = app_module.app.test_client()
    image_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='green').save(image_bytes, format='PNG')
    image_bytes.seek(0)

    analyze_resp = client.post('/api/analyze', data={'image': (image_bytes, 'sample.png'), 'lang': 'en'}, content_type='multipart/form-data')
    assert analyze_resp.status_code == 200

    history_resp = client.get('/api/predictions?limit=5')
    assert history_resp.status_code == 200
    payload = history_resp.get_json()
    assert payload['predictions']
    assert payload['predictions'][0]['class_name']


def test_prediction_history_export_and_analytics(monkeypatch):
    _use_ungated_analysis(monkeypatch)
    app_module.engine.dispose()
    app_module.Base.metadata.drop_all(bind=app_module.engine)
    app_module.Base.metadata.create_all(bind=app_module.engine)

    client = app_module.app.test_client()
    image_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='green').save(image_bytes, format='PNG')
    image_bytes.seek(0)

    analyze_resp = client.post('/api/analyze', data={'image': (image_bytes, 'sample.png'), 'lang': 'en'}, content_type='multipart/form-data')
    assert analyze_resp.status_code == 200

    export_resp = client.get('/api/predictions/export')
    assert export_resp.status_code == 200
    assert 'class_name' in export_resp.get_data(as_text=True)

    analytics_resp = client.get('/api/predictions/analytics')
    assert analytics_resp.status_code == 200
    payload = analytics_resp.get_json()
    assert payload['series']
