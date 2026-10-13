"""Regression suite for the binary leaf / non-leaf classifier.

Categories:
  A) healthy PlantVillage leaves must classify as LEAF
  B) diseased leaves the gate already accepts must classify as LEAF
  C) images documented as UNCERTAIN must NOT be converted to LEAF.
     This suite deliberately does not assert that every field image is
     accepted: ambiguous photos belong in the UNCERTAIN band, and
     tightening them into LEAF is exactly the regression to catch.
  D) hard-negative non-leaf fixtures must never classify as LEAF.

Fixtures come from experiments/leaf_classifier/regression_fixture_selection.json
plus the vehicle hard negative in artifacts/leaf_classifier/dataset/nonleaf/vehicle.
Paths and thresholds are never hard-coded: paths are derived from the fixture
selection file relative to the repo root, and every threshold comparison reads
the loaded classifier's own config-backed attributes, so a legitimate
threshold retune does not break this suite.

Test-only: this module imports leaf_classifier, runs inference, and asserts
on verdicts. It must not modify any model, config, dataset, or production
artifact, and it never writes anywhere.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import leaf_classifier  # noqa: E402

FIXTURE_SELECTION = Path(__file__).resolve().parent / 'regression_fixture_selection.json'
NONLEAF_VEHICLE_DIR = ROOT / 'artifacts' / 'leaf_classifier' / 'dataset' / 'nonleaf' / 'vehicle'


def _load_fixtures():
    if not FIXTURE_SELECTION.exists():
        return None
    return json.loads(FIXTURE_SELECTION.read_text(encoding='utf-8-sig'))


_FIXTURES = _load_fixtures()


def _fixture_params(section):
    """Param list for a fixture-selection section, or a clear skip sentinel."""
    params = []
    if _FIXTURES:
        for item in _FIXTURES.get(section, []):
            path = ROOT / item['path']
            if 'name' in item:
                id_ = item['name']
            else:
                id_ = '{}::{}'.format(item['class_dir'], os.path.basename(item['path']))
            params.append(pytest.param(str(path), id=id_))
    if not params:
        params.append(pytest.param(None, id='fixture-selection-missing'))
    return params


def _nonleaf_params():
    """Documented non-leaf hard negatives plus every vehicle fixture on disk."""
    params = _fixture_params('nonleaf')
    if params and params[0].values[0] is None:
        params = []
    vehicle_files = sorted(NONLEAF_VEHICLE_DIR.glob('*.jpg')) if NONLEAF_VEHICLE_DIR.is_dir() else []
    for path in vehicle_files:
        params.append(pytest.param(str(path), id='vehicle::{}'.format(path.name)))
    if not params:
        params.append(pytest.param(None, id='nonleaf-fixtures-missing'))
    return params


@pytest.fixture(scope='module')
def clf():
    classifier = leaf_classifier.load()
    if classifier is None:
        pytest.skip('leaf classifier unavailable (checkpoint or config missing)')
    return classifier


def _run(clf, path):
    if path is None:
        pytest.skip('fixture genuinely unavailable')
    if not os.path.exists(path):
        pytest.skip('fixture missing on disk: {}'.format(path))
    image = Image.open(path).convert('RGB')
    return clf.classify(image)


# ---- Category A: healthy leaves -------------------------------------------

@pytest.mark.parametrize('path', _fixture_params('healthy'))
def test_category_a_healthy_leaf_is_leaf(clf, path):
    """A healthy PlantVillage leaf must be accepted as LEAF."""
    verdict = _run(clf, path)
    assert verdict['state'] == leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is True
    assert verdict['leaf_probability'] >= clf.accept_threshold


# ---- Category B: diseased leaves already accepted --------------------------

@pytest.mark.parametrize('path', _fixture_params('diseased'))
def test_category_b_diseased_leaf_is_leaf(clf, path):
    """A diseased leaf the gate already accepts must stay LEAF."""
    verdict = _run(clf, path)
    assert verdict['state'] == leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is True
    assert verdict['leaf_probability'] >= clf.accept_threshold


# ---- Category C: documented UNCERTAIN images -------------------------------

@pytest.mark.parametrize('path', _fixture_params('ambiguous'))
def test_category_c_ambiguous_not_converted_to_leaf(clf, path):
    """Images documented as UNCERTAIN must not regress into LEAF.

    The bound is the accept threshold from the classifier's own config, not
    a hard-coded number: as long as the score stays below accept, the image
    is refused or asked-for-again, never diagnosed.
    """
    verdict = _run(clf, path)
    assert verdict['state'] != leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is not True
    assert verdict['leaf_probability'] < clf.accept_threshold


# ---- Category D: hard-negative non-leaf fixtures ---------------------------

@pytest.mark.parametrize('path', _nonleaf_params())
def test_category_d_non_leaf_never_leaf(clf, path):
    """A non-leaf photograph must never classify as LEAF.

    Hard negatives may land in UNCERTAIN (ask for a clearer photo) or
    NON_LEAF (refuse outright); either is acceptable, LEAF is not.
    """
    verdict = _run(clf, path)
    assert verdict['state'] != leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is not True
    assert verdict['leaf_probability'] < clf.accept_threshold


# ---- Contract tests --------------------------------------------------------

def test_thresholds_come_from_config_and_are_ordered(clf):
    """accept must exceed reject, both loaded from the config artifact."""
    assert clf.accept_threshold > clf.reject_threshold


def test_classify_never_raises_for_readable_image(clf):
    """classify() must return a verdict dict, never raise (fail closed)."""
    path = None
    if _FIXTURES and _FIXTURES.get('healthy'):
        path = str(ROOT / _FIXTURES['healthy'][0]['path'])
    elif _FIXTURES and _FIXTURES.get('nonleaf'):
        path = str(ROOT / _FIXTURES['nonleaf'][0]['path'])
    if path is None or not os.path.exists(path):
        pytest.skip('no readable fixture available')
    image = Image.open(path).convert('RGB')
    verdict = clf.classify(image)
    assert verdict['state'] in (leaf_classifier.STATE_LEAF,
                                leaf_classifier.STATE_UNCERTAIN,
                                leaf_classifier.STATE_NON_LEAF)
    assert 'leaf_probability' in verdict
