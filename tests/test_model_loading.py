import os
import sys
import tempfile

import torch
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_model.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module


def test_load_model_accepts_model_state_dict_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = models.mobilenet_v2(weights=None, num_classes=len(app_module.idx_to_class))
        checkpoint_path = os.path.join(tmpdir, 'model_checkpoint.pt')
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

        loaded = app_module.load_model(checkpoint_path)

        assert loaded is not None
        assert loaded.__class__.__name__ == 'MobileNetV2'
