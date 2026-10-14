"""Regression test: REMEDY_MAP must cover every authoritative class name.

Root cause this guards against: app.py's REMEDY_MAP used legacy class-name
spellings (e.g. `Corn___Common_rust`, `Orange___Haunglongbing__Citrus_greening_`)
that do not match the authoritative names in index_to_name.json. For any class
whose name was not a literal key, REMEDY_MAP.get() returned None and the
request fell through to the advisory LLM chain (NVIDIA -> OpenRouter -> Gemini),
which is the dominant contributor to the 10-18s /api/analyze times reported by
mobile users. The fix is a key-only change; no class, confidence, top-k,
Grad-CAM or leaf-gate behaviour is affected.

This test derives the authoritative names from index_to_name.json at runtime,
so it cannot drift from the production mapping.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module  # noqa: E402


def _authoritative_names():
    with open(os.path.join(ROOT, 'index_to_name.json'), 'r') as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def test_remedy_map_covers_every_authoritative_class():
    names = _authoritative_names()
    missing = [names[i] for i in range(len(names))
               if names[i] not in app_module.REMEDY_MAP]
    assert not missing, (
        'REMEDY_MAP is missing {} authoritative class name(s): {}. '
        'Every prediction of those classes falls through to the advisory LLM '
        'chain and can take 10-18s. Add the authoritative name as a literal '
        'key (the remedy text may be copied from the legacy key).'.format(
            len(missing), missing))


def test_remedy_map_has_no_legacy_spelling_leftover():
    """The 12 legacy spellings that used to be keys must no longer be present."""
    legacy = {
        'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
        'Corn___Common_rust',
        'Corn___Northern_Leaf_Blight',
        'Corn___healthy',
        'Cherry___Powdery_mildew',
        'Cherry___healthy',
        'Grape___Esca__Black_Measles_',
        'Grape___Leaf_blight__Isariopsis_Leaf_Spot_',
        'Orange___Haunglongbing__Citrus_greening_',
        'Pepper__bell___Bacterial_spot',
        'Pepper__bell___healthy',
        'Tomato___Spider_mites_Two-spotted_spider_mite',
    }
    leftover = sorted(legacy & set(app_module.REMEDY_MAP))
    assert not leftover, (
        'Legacy REMEDY_MAP keys still present: {}. They do not match any '
        'authoritative class name and would silently fall through to the '
        'advisory LLM chain.'.format(leftover))


def test_remedy_map_size_matches_class_count():
    names = _authoritative_names()
    assert len(app_module.REMEDY_MAP) == len(names), (
        'REMEDY_MAP has {} entries but there are {} authoritative '
        'classes'.format(len(app_module.REMEDY_MAP), len(names)))