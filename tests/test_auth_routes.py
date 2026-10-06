import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_auth.db')
os.environ.setdefault('FLASK_TESTING', '1')

for path in ['test_auth.db', 'test_auth.db-journal']:
    if os.path.exists(os.path.join(ROOT, path)):
        os.remove(os.path.join(ROOT, path))

import app as app_module


def test_register_and_login():
    app_module.engine.dispose()
    app_module.Base.metadata.drop_all(bind=app_module.engine)
    app_module.Base.metadata.create_all(bind=app_module.engine)
    client = app_module.app.test_client()

    register_resp = client.post('/api/auth/register', json={
        'username': 'farmer1',
        'password': 'secret123'
    })
    assert register_resp.status_code == 200
    register_data = register_resp.get_json()
    assert register_data['token']
    assert register_data['user']['username'] == 'farmer1'

    login_resp = client.post('/api/auth/login', json={
        'username': 'farmer1',
        'password': 'secret123'
    })
    assert login_resp.status_code == 200
    login_data = login_resp.get_json()
    assert login_data['token']
