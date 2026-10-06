import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault('OPENROUTER_API_KEY', 'test')
os.environ.setdefault('OPENWEATHER_API_KEY', 'test')
os.environ.setdefault('DATABASE_URL', 'sqlite:///test_advisory_budget.db')
os.environ.setdefault('FLASK_TESTING', '1')

import app as app_module


class _SlowProvider:
    """Stand-in for an OpenAI-compatible client whose endpoint never answers."""

    def __init__(self, delay=30.0, content=None):
        self.delay = delay
        self.content = content
        self.timeouts = []
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.timeouts.append(kwargs.get('timeout'))
        if self.content is not None:
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content))])
        timeout = kwargs.get('timeout')
        time.sleep(float(timeout) if timeout else self.delay)
        raise RuntimeError('provider unavailable')


def _only(monkeypatch, *providers):
    svc = app_module.ai_service
    monkeypatch.setattr(svc, 'nvidia_client', None, raising=False)
    monkeypatch.setattr(svc, 'openrouter_client', None, raising=False)
    monkeypatch.setattr(svc, 'gemini_model', None, raising=False)
    for name, provider in zip(('nvidia_client', 'openrouter_client'), providers):
        if provider is not None:
            monkeypatch.setattr(svc, name, provider, raising=False)
    return svc


def test_healthy_provider_still_answers_within_budget(monkeypatch):
    monkeypatch.setattr(app_module, 'ADVISORY_LLM_BUDGET_S', 8.0)
    provider = _SlowProvider(delay=0.0, content='apply a labelled fungicide')
    svc = _only(monkeypatch, provider)

    content, error = svc.generate_ai_response(
        [{'role': 'user', 'content': 'Corn_(maize)___Common_rust_'}])

    assert content == 'apply a labelled fungicide'
    assert error is None
    assert provider.timeouts == [8.0]


def test_unhealthy_chain_is_capped_by_the_budget(monkeypatch):
    monkeypatch.setattr(app_module, 'ADVISORY_LLM_BUDGET_S', 1.0)
    first = _SlowProvider()
    second = _SlowProvider()
    svc = _only(monkeypatch, first, second)

    start = time.perf_counter()
    content, error = svc.generate_ai_response(
        [{'role': 'user', 'content': 'Corn_(maize)___Common_rust_'}])
    elapsed = time.perf_counter() - start

    assert content is None
    assert error, 'must fall back to local knowledge when providers fail'
    assert elapsed < 1.0 + 4.0, f'chain took {elapsed:.2f}s, budget was 1s'
    assert first.timeouts == [1.0]
    for timeout in second.timeouts:
        assert timeout <= 1.0


def test_exhausted_budget_skips_every_provider(monkeypatch):
    monkeypatch.setattr(app_module, 'ADVISORY_LLM_BUDGET_S', 0.0)
    provider = _SlowProvider(delay=0.0, content='never used')
    svc = _only(monkeypatch, provider)

    content, error = svc.generate_ai_response(
        [{'role': 'user', 'content': 'Corn_(maize)___Common_rust_'}])

    assert content is None
    assert error
    assert provider.calls == 0