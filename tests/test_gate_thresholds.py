"""Issue 1 regression: gate rejection must report the DECIDING validator's
thresholds, not the heuristic gate's.

app._gate_rejection_response used to read LEAF_GATE.accept_threshold /
LEAF_GATE.reject_threshold unconditionally, so a LEAF_UNCERTAIN verdict
produced by the leaf classifier (accept 0.62 / reject 0.10) was explained
with the gate's thresholds (0.90 / 0.05). The frontend reads those fields to
explain the verdict, so it reported a threshold the deciding validator never
used.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_gate_thresholds.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module


def test_gate_rejection_reports_classifier_thresholds_for_classifier_verdict():
    verdict = {
        'state': 'UNCERTAIN', 'is_leaf': None,
        'leaf_probability': 0.114946,
        'reason': 'inconclusive',
        'accept_threshold': 0.62,
        'reject_threshold': 0.10,
    }
    decision = app_module._classifier_verdict_to_decision(verdict)
    response = app_module._gate_rejection_response(decision, 'en', 'upload')
    gate = response['leaf_gate']
    assert gate['accept_threshold'] == 0.62, (
        'classifier verdict must report the classifier accept threshold, got %s'
        % gate['accept_threshold'])
    assert gate['reject_threshold'] == 0.10, (
        'classifier verdict must report the classifier reject threshold, got %s'
        % gate['reject_threshold'])
    assert gate['signals']['validator'] == 'leaf_classifier'


def test_gate_rejection_reports_gate_thresholds_for_gate_verdict():
    decision = {
        'state': 'UNCERTAIN', 'is_leaf': None,
        'leaf_confidence': 0.5,
        'reason': 'weak evidence',
        'score_basis': 'vegetation+feature+coherence+detail',
        'signals': {'validator': 'heuristic_gate'},
    }
    response = app_module._gate_rejection_response(decision, 'en', 'upload')
    gate = response['leaf_gate']
    assert gate['accept_threshold'] == app_module.LEAF_GATE.accept_threshold
    assert gate['reject_threshold'] == app_module.LEAF_GATE.reject_threshold


def test_classifier_verdict_carries_its_own_thresholds():
    clf = getattr(app_module, 'LEAF_CLASSIFIER', None)
    if clf is None:
        pytest.skip('leaf classifier not loaded')
    assert clf.accept_threshold == 0.62
    assert clf.reject_threshold == 0.10