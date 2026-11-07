"""Tests for nanochat/chat_service.py with torch and nanochat stubbed out."""
import asyncio
import importlib.util
import os
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def svc(monkeypatch):
    stubs = {name: mock.MagicMock() for name in (
        "torch", "nanochat", "nanochat.common",
        "nanochat.checkpoint_manager", "nanochat.engine")}
    stubs["nanochat.common"].compute_init.return_value = (False, 0, 0, 1, "cpu")
    stubs["nanochat.common"].autodetect_device_type.return_value = "cpu"
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "argv", ["chat_service.py"])
    spec = importlib.util.spec_from_file_location(
        "chat_service", os.path.join(ROOT, "nanochat", "chat_service.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool(svc, generate):
    tokenizer = mock.MagicMock()
    tokenizer.get_bos_token_id.return_value = 0
    tokenizer.encode_special.side_effect = lambda s: {"<|assistant_end|>": 9}.get(s, 1)
    tokenizer.encode.return_value = [5, 6]
    tokenizer.decode.side_effect = lambda toks: "x" * len(toks)
    engine = mock.MagicMock()
    engine.generate.side_effect = generate
    worker = svc.Worker(gpu_id=0, device="cpu", engine=engine,
                        tokenizer=tokenizer, autocast_ctx=mock.MagicMock())
    pool = svc.WorkerPool(num_gpus=1)
    pool.workers.append(worker)
    pool.available_workers.put_nowait(worker)
    return pool


def _post(svc, pool, stream=False):
    svc.app.state.worker_pool = pool
    client = TestClient(svc.app, raise_server_exceptions=False)
    return client.post("/v1/chat/completions", json={
        "model": "nanochat", "stream": stream,
        "messages": [{"role": "user", "content": "hi"}]})


def test_non_streaming_success_releases_worker_once(svc):
    pool = _pool(svc, lambda *a, **k: iter([([7], [1]), ([7], [1]), ([9], [1])]))
    r = _post(svc, pool)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "xx"
    assert pool.available_workers.qsize() == 1


def test_generation_error_releases_worker_once(svc):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
        yield
    pool = _pool(svc, boom)
    assert _post(svc, pool).status_code == 500
    # A double release would put the same worker in the queue twice, letting
    # two requests run concurrently on one GPU replica.
    assert pool.available_workers.qsize() == 1


def test_streaming_releases_worker_once(svc):
    pool = _pool(svc, lambda *a, **k: iter([([7], [1]), ([9], [1])]))
    r = _post(svc, pool, stream=True)
    assert r.status_code == 200
    assert r.text.rstrip().endswith("data: [DONE]")
    assert pool.available_workers.qsize() == 1
