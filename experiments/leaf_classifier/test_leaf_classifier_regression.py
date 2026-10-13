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
plus the vehicle hard negatives in
artifacts/leaf_classifier/dataset/nonleaf/vehicle. Paths and thresholds are
never hard-coded: paths are derived from the fixture selection file relative
to the repo root, and every threshold comparison reads the loaded
classifier's own config-backed attributes, so a legitimate threshold retune
does not break this suite.

Portability: the manifest references local datasets that are ignored by Git
(data/PlantVillage, validation/, and the curated non-leaf collection under
artifacts/leaf_classifier/dataset/nonleaf), and the suite also needs the
classifier checkpoint and config under artifacts/leaf_classifier/. This
suite is therefore NOT portable to a fresh clone on its own. When a
required fixture or artifact is unavailable, the affected tests SKIP with
a message naming the missing category and the expected path; a missing
fixture category is never counted as a passing test.

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

CATEGORY_A = 'A: healthy leaves'
CATEGORY_B = 'B: diseased leaves'
CATEGORY_C = 'C: documented UNCERTAIN images'
CATEGORY_D = 'D: non-leaf hard negatives'


def _load_fixtures():
    if not FIXTURE_SELECTION.exists():
        return None
    return json.loads(FIXTURE_SELECTION.read_text(encoding='utf-8-sig'))


_FIXTURES = _load_fixtures()


def _section_params(section, category):
    """(category, path, expected) params for one manifest section.

    Each param carries the fixture category, the absolute fixture path,
    and the human-readable location a user should look for. When the
    manifest or the section is unavailable a single diagnostic sentinel
    is emitted instead, so the category is reported as SKIPPED - with
    the expected path in the skip reason - rather than silently
    collected as zero tests.
    """
    params = []
    if _FIXTURES:
        for item in _FIXTURES.get(section, []):
            path = ROOT / item['path']
            if 'name' in item:
                id_ = item['name']
            else:
                id_ = '{}::{}'.format(item['class_dir'],
                                      os.path.basename(item['path']))
            params.append(pytest.param(category, str(path), str(path),
                                       id=id_))
    if not params:
        params.append(pytest.param(
            category, None, str(FIXTURE_SELECTION),
            id='{}-fixture-unavailable'.format(section)))
    return params


def _nonleaf_params():
    """Documented non-leaf hard negatives plus every vehicle fixture on disk."""
    has_documented = bool(_FIXTURES and _FIXTURES.get('nonleaf'))
    params = _section_params('nonleaf', CATEGORY_D) if has_documented else []
    if NONLEAF_VEHICLE_DIR.is_dir():
        for path in sorted(NONLEAF_VEHICLE_DIR.glob('*.jpg')):
            params.append(pytest.param(
                CATEGORY_D, str(path), str(path),
                id='vehicle::{}'.format(path.name)))
    if not params:
        params.append(pytest.param(
            CATEGORY_D, None,
            '"nonleaf" section of {} or .jpg files in {}'.format(
                FIXTURE_SELECTION, NONLEAF_VEHICLE_DIR),
            id='nonleaf-fixtures-unavailable'))
    return params


@pytest.fixture(scope='module')
def clf():
    classifier = leaf_classifier.load()
    if classifier is None:
        pytest.skip(
            'classifier unavailable - expected checkpoint {} and config {}'
            .format(leaf_classifier.DEFAULT_CHECKPOINT,
                    leaf_classifier.DEFAULT_CONFIG))
    return classifier


def _open_fixture(category, path, expected):
    """Open a fixture image, or skip naming the category and expected path."""
    if path is None:
        pytest.skip('{}: fixture unavailable - expected {}'.format(
            category, expected))
    if not os.path.exists(path):
        pytest.skip('{}: fixture not found on disk - expected {} '
                    '(local dataset, gitignored)'.format(category, path))
    return Image.open(path).convert('RGB')


# ---- Category A: healthy leaves -------------------------------------------

@pytest.mark.parametrize('category,path,expected',
                         _section_params('healthy', CATEGORY_A))
def test_category_a_healthy_leaf_is_leaf(clf, category, path, expected):
    """A healthy PlantVillage leaf must be accepted as LEAF."""
    image = _open_fixture(category, path, expected)
    verdict = clf.classify(image)
    assert verdict['state'] == leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is True
    assert verdict['leaf_probability'] >= clf.accept_threshold


# ---- Category B: diseased leaves already accepted --------------------------

@pytest.mark.parametrize('category,path,expected',
                         _section_params('diseased', CATEGORY_B))
def test_category_b_diseased_leaf_is_leaf(clf, category, path, expected):
    """A diseased leaf the gate already accepts must stay LEAF."""
    image = _open_fixture(category, path, expected)
    verdict = clf.classify(image)
    assert verdict['state'] == leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is True
    assert verdict['leaf_probability'] >= clf.accept_threshold


# ---- Category C: documented UNCERTAIN images -------------------------------

@pytest.mark.parametrize('category,path,expected',
                         _section_params('ambiguous', CATEGORY_C))
def test_category_c_ambiguous_not_converted_to_leaf(clf, category, path,
                                                    expected):
    """Images documented as UNCERTAIN must not regress into LEAF.

    The bound is the accept threshold from the classifier's own config, not
    a hard-coded number: as long as the score stays below accept, the image
    is refused or asked-for-again, never diagnosed.
    """
    image = _open_fixture(category, path, expected)
    verdict = clf.classify(image)
    assert verdict['state'] != leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is not True
    assert verdict['leaf_probability'] < clf.accept_threshold


# ---- Category D: hard-negative non-leaf fixtures ---------------------------

@pytest.mark.parametrize('category,path,expected', _nonleaf_params())
def test_category_d_non_leaf_never_leaf(clf, category, path, expected):
    """A non-leaf photograph must never classify as LEAF.

    Hard negatives may land in UNCERTAIN (ask for a clearer photo) or
    NON_LEAF (refuse outright); either is acceptable, LEAF is not.
    """
    image = _open_fixture(category, path, expected)
    verdict = clf.classify(image)
    assert verdict['state'] != leaf_classifier.STATE_LEAF
    assert verdict['is_leaf'] is not True
    assert verdict['leaf_probability'] < clf.accept_threshold


# ---- Contract tests --------------------------------------------------------

def test_thresholds_come_from_config_and_are_ordered(clf):
    """accept must exceed reject, both loaded from the config artifact."""
    assert clf.accept_threshold > clf.reject_threshold


def test_classify_never_raises_for_readable_image(clf):
    """classify() must return a verdict dict, never raise (fail closed)."""
    category, path = None, None
    if _FIXTURES and _FIXTURES.get('healthy'):
        category, path = (CATEGORY_A,
                          str(ROOT / _FIXTURES['healthy'][0]['path']))
    elif _FIXTURES and _FIXTURES.get('nonleaf'):
        category, path = (CATEGORY_D,
                          str(ROOT / _FIXTURES['nonleaf'][0]['path']))
    if path is None:
        pytest.skip(
            'contract check needs any readable fixture - expected the '
            '"healthy" or "nonleaf" section of {}'.format(FIXTURE_SELECTION))
    if not os.path.exists(path):
        pytest.skip('{}: fixture not found on disk - expected {} '
                    '(local dataset, gitignored)'.format(category, path))
    image = Image.open(path).convert('RGB')
    verdict = clf.classify(image)
    assert verdict['state'] in (leaf_classifier.STATE_LEAF,
                                leaf_classifier.STATE_UNCERTAIN,
                                leaf_classifier.STATE_NON_LEAF)
    assert 'leaf_probability' in verdict
