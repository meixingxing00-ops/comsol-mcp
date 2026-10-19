"""Tests for the runtime-lock and abandoned-call guards (no COMSOL needed)."""

import concurrent.futures
import json
import threading

import pytest

import comsol_mcp._server as srv
from comsol_mcp._connection import _require_no_abandoned_call
from comsol_mcp._state import _run_tool_model_read


def test_model_read_fails_fast_while_lock_is_held():
    held = threading.Event()
    release = threading.Event()

    def holder():
        with srv._runtime_lock:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait(5)
    try:
        payload = json.loads(_run_tool_model_read("model_tree", lambda: {"never": "called"}))
    finally:
        release.set()
        thread.join()
    assert payload["success"] is False
    assert "run_study_status" in payload["error"]


def test_model_read_runs_when_lock_is_free():
    payload = json.loads(_run_tool_model_read("model_tree", lambda: {"ok": 1}))
    assert payload["success"] is True


def test_abandoned_call_blocks_until_done(monkeypatch):
    future = concurrent.futures.Future()
    monkeypatch.setattr(srv, "_abandoned_calls", [("Loading model x.mph timed out", future)])
    with pytest.raises(RuntimeError, match="still running"):
        _require_no_abandoned_call()
    future.set_result(None)
    _require_no_abandoned_call()
    assert srv._abandoned_calls == []
