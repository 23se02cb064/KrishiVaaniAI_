import io
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_batch.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module


def test_batch_analyze_accepts_multiple_images(monkeypatch):
    # Batch mechanics, not leaf suitability: the flat green stand-in is refused by the
    # leaf validator, which is correct behaviour. See tests/test_leaf_gate.py for the
    # gate. LEAF_VALIDATION_BYPASS is the explicit escape hatch for exercising the
    # unguarded plumbing path; the default refuses, which is why it is set here.
    monkeypatch.setattr(app_module, 'LEAF_VALIDATION_BYPASS', True)
    client = app_module.app.test_client()
    image_bytes = io.BytesIO()
    Image.new('RGB', (224, 224), color='green').save(image_bytes, format='PNG')
    image_bytes.seek(0)

    resp = client.post('/api/analyze-batch', data={
        'images': [(image_bytes, 'sample.png')]
    }, content_type='multipart/form-data')

    assert resp.status_code == 200
    payload = resp.get_json()
    assert 'results' in payload
    assert len(payload['results']) == 1
    assert payload['results'][0]['class']
