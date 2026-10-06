"""Grad-CAM must produce a real heatmap, not the failure placeholder.

gradcam.generate_gradcam_overlay catches every exception and returns a solid-gray PNG
placeholder instead of raising, so a broken Grad-CAM is invisible to a truthiness check:
the placeholder is valid, non-empty base64 and `bool(overlay)` is True. The API kept
returning 200 with `gradcam_overlay` set, and a report that recorded
"all_gradcam_overlays_produced: True" was accurate about the field and blind to the
content.

That is exactly what happened. A `torch.set_grad_enabled(False)` call in
leaf_classifier.py disabled autograd process-wide, so every `output.backward()` in
gradcam.py raised "element 0 of tensors does not require grad" and every overlay
degraded to gray. These tests assert the overlay has real content, and that the
process-wide autograd state is intact.
"""

import base64
import io
import os
import sys
from io import BytesIO

import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_gradcam.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module  # noqa: E402
import gradcam  # noqa: E402

PLANTVILLAGE_VAL = os.path.join(ROOT, 'data', 'PlantVillage', 'val')


def _first_leaf_image():
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


def _colour_count(overlay_b64):
    """Number of distinct colours, or None when there are more than the cap.

    getcolors() returns None to mean "more than maxcolors", which is itself proof of a
    real overlay; only a small count indicates the flat placeholder.
    """
    image = Image.open(BytesIO(base64.b64decode(overlay_b64))).convert('RGB')
    return image.getcolors(maxcolors=4096)


def test_importing_app_does_not_disable_autograd():
    """A module-level torch.set_grad_enabled(False) anywhere breaks Grad-CAM.

    autograd state is process-wide, so this is a whole-application invariant, not a
    property of the leaf classifier alone.
    """
    assert torch.is_grad_enabled(), (
        'autograd is disabled process-wide; Grad-CAM backward passes cannot run'
    )


def test_gradcam_overlay_has_real_content():
    path = _first_leaf_image()
    if path is None:
        pytest.skip('PlantVillage validation split not present on this machine')

    with open(path, 'rb') as handle:
        overlay, heatmap, class_idx, error = gradcam.generate_gradcam_overlay(
            app_module.model, handle.read())

    assert error is None, f'Grad-CAM reported an error: {error}'
    assert class_idx >= 0, f'no class was resolved (class_idx={class_idx})'

    colours = _colour_count(overlay)
    assert colours is None or len(colours) > 8, (
        'overlay is a flat placeholder rather than a heatmap blended onto the image')

    heatmap_colours = _colour_count(heatmap)
    assert heatmap_colours is None or len(heatmap_colours) > 4, (
        'heatmap is flat, so no attention map was computed')


def test_leaf_classifier_inference_does_not_leak_grad_state():
    """Running the leaf classifier must leave grad mode exactly as it found it.

    The classifier only needs inference, but disabling autograd for that is the
    classifier's business alone. A global toggle is not: it also removes Grad-CAM.
    """
    classifier = getattr(app_module, 'LEAF_CLASSIFIER', None)
    if classifier is None:
        pytest.skip('leaf classifier not loaded')

    before = torch.is_grad_enabled()
    path = _first_leaf_image()
    if path is None:
        pytest.skip('PlantVillage validation split not present on this machine')

    image = Image.open(path).convert('RGB')
    classifier.probability(image)

    assert torch.is_grad_enabled() == before, (
        'leaf classifier changed the global grad-enabled state')


def test_api_returns_a_real_gradcam_overlay_for_a_diagnosed_leaf():
    path = _first_leaf_image()
    if path is None:
        pytest.skip('PlantVillage validation split not present on this machine')

    client = app_module.app.test_client()
    with open(path, 'rb') as handle:
        response = client.post('/api/analyze', data={
            'image': (io.BytesIO(handle.read()), os.path.basename(path)),
            'include_gradcam': 'true',
        }, content_type='multipart/form-data')

    payload = response.get_json() or {}
    if response.status_code != 200:
        pytest.skip(f'image was not diagnosed ({response.status_code}), nothing to check')

    assert not payload.get('gradcam_error'), payload.get('gradcam_error')
    overlay = payload.get('gradcam_overlay')
    assert overlay, 'no overlay returned'

    colours = _colour_count(overlay)
    assert colours is None or len(colours) > 8, (
        'API returned the gray Grad-CAM placeholder while reporting no error')
