"""Tests for pure helper functions that don't need COMSOL."""

import os
import json
from pathlib import Path

import pytest

from comsol_mcp._state import (
    _sanitize_snapshot_label, _port_is_open, _default_workflow_state,
    _friendly_connection_error, _resolve_path, _resolve_output_path,
    _json, _now_iso, _read_workflow_state, _write_workflow_state,
)
from comsol_mcp._model_ops import (
    _normalize_properties, _coerce_eval_value, _last_scalar,
)
from comsol_mcp._model import _normcase_path


class TestSanitizeSnapshotLabel:
    def test_alphanumeric_passthrough(self):
        assert _sanitize_snapshot_label("baseline_v3") == "baseline_v3"

    def test_special_characters_replaced(self):
        assert _sanitize_snapshot_label("trial eta=0.75") == "trial_eta_0_75"

    def test_empty_returns_snapshot(self):
        assert _sanitize_snapshot_label("") == "snapshot"

    def test_only_underscores_trimmed(self):
        assert _sanitize_snapshot_label("___") == "snapshot"

    def test_dots_and_dashes_preserved(self):
        assert _sanitize_snapshot_label("run-1_test") == "run-1_test"


class TestNormalizeProperties:
    def test_empty_string(self):
        assert _normalize_properties("") == []

    def test_json_object(self):
        result = _normalize_properties('{"size": "0.1"}')
        assert result == [("size", ["0.1"])]

    def test_json_object_multi_value(self):
        result = _normalize_properties('{"size": ["0.1", "0.2"]}')
        assert result == [("size", ["0.1", "0.2"])]

    def test_json_array(self):
        result = _normalize_properties('[{"name": "pos", "value": "1"}]')
        assert result == [("pos", ["1"])]

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            _normalize_properties("42")


class TestCoerceEvalValue:
    def test_scalar(self):
        assert _coerce_eval_value(3.14) == 3.14

    def test_string(self):
        assert _coerce_eval_value("hello") == "hello"

    def test_none(self):
        assert _coerce_eval_value(None) is None

    def test_nested_list(self):
        assert _coerce_eval_value([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_tolist_object(self):
        class FakeArray:
            def tolist(self):
                return [1, 2, 3]
        assert _coerce_eval_value(FakeArray()) == [1, 2, 3]


class TestLastScalar:
    def test_scalar(self):
        assert _last_scalar(42) == 42

    def test_single_element(self):
        assert _last_scalar([42]) == 42

    def test_nested_list(self):
        assert _last_scalar([[1, 2], [3, 4]]) == 4

    def test_empty_list(self):
        assert _last_scalar([]) == []


class TestPortIsOpen:
    def test_closed_port(self):
        assert _port_is_open("localhost", 1, timeout_seconds=0.1) is False


class TestDefaultWorkflowState:
    def test_has_required_keys(self):
        state = _default_workflow_state()
        required = [
            "mode", "current_main_model_path", "visible_main_locked",
            "guard_level", "workflow_stage", "main_model_tag",
            "main_model_label", "main_model_path",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_default_values(self):
        state = _default_workflow_state()
        assert state["visible_main_locked"] is False
        assert state["guard_level"] == "strict"
        assert state["workflow_stage"] == "awaiting_manual_server"


class TestFriendlyConnectionError:
    def test_connection_refused(self):
        exc = ConnectionError("Connection was actively refused")
        result = _friendly_connection_error(exc, "localhost", 2036)
        assert "refused" in str(result).lower()

    def test_auth_failure(self):
        exc = Exception("Authentication required: invalid password")
        result = _friendly_connection_error(exc, "localhost", 2036)
        assert "authentication" in str(result).lower() or "auth" in str(result).lower()

    def test_generic_error(self):
        exc = RuntimeError("Something weird happened")
        result = _friendly_connection_error(exc, "localhost", 2036)
        assert "2036" in str(result)


class TestNormcasePath:
    def test_empty(self):
        assert _normcase_path("") == ""

    def test_absolute_path(self):
        result = _normcase_path("C:/Users/test/model.mph")
        assert "model.mph" in result.lower()


class TestJsonHelper:
    def test_basic(self):
        result = _json({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result


class TestNowIso:
    def test_format(self):
        result = _now_iso()
        assert len(result) == 19  # YYYY-MM-DD HH:MM:SS
        assert "T" not in result


class TestResolvePath:
    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_path(str(tmp_path / "nonexistent.mph"))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _resolve_path("")

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.mph"
        f.write_text("test")
        result = _resolve_path(str(f))
        assert result.exists()


class TestWorkflowStateIO:
    def test_round_trip(self, tmp_path, monkeypatch):
        import comsol_mcp._state as state_mod
        monkeypatch.setattr(state_mod, "WORKFLOW_FILE", tmp_path / "workflow_state.json")
        monkeypatch.setattr(state_mod, "COMSOL_SERVER_MCP_HOME", tmp_path)
        monkeypatch.setattr(state_mod, "LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr(state_mod, "OUTPUTS_DIR", tmp_path / "outputs")

        _write_workflow_state({"mode": "test-mode", "notes": "hello"})
        state = _read_workflow_state()
        assert state["mode"] == "test-mode"
        assert state["notes"] == "hello"

    def test_corrupted_json_fallback(self, tmp_path, monkeypatch):
        import comsol_mcp._state as state_mod
        wf = tmp_path / "workflow_state.json"
        wf.write_text("{invalid json", encoding="utf-8")
        monkeypatch.setattr(state_mod, "WORKFLOW_FILE", wf)
        monkeypatch.setattr(state_mod, "COMSOL_SERVER_MCP_HOME", tmp_path)

        state = _read_workflow_state()
        assert state["mode"] == "manual"  # falls back to default


class TestBackupExistingFile:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        import comsol_mcp._model as model_mod
        monkeypatch.setattr(model_mod, "OUTPUTS_DIR", tmp_path / "outputs")
        assert model_mod._backup_existing_file(tmp_path / "absent.mph") == ""

    def test_existing_file_is_copied(self, tmp_path, monkeypatch):
        import comsol_mcp._model as model_mod
        monkeypatch.setattr(model_mod, "OUTPUTS_DIR", tmp_path / "outputs")
        original = tmp_path / "main.mph"
        original.write_bytes(b"original")
        backup = Path(model_mod._backup_existing_file(original))
        assert backup.parent == tmp_path / "outputs" / "backups"
        assert backup.name.startswith("main_") and backup.suffix == ".mph"
        assert backup.read_bytes() == b"original"
        assert original.read_bytes() == b"original"


class TestExpressionItems:
    def test_parse_rejects_non_array(self):
        from comsol_mcp._model_ops import _parse_expression_items
        with pytest.raises(ValueError):
            _parse_expression_items('{"name": "a"}', "core_metrics_json")

    def test_parse_requires_expression(self):
        from comsol_mcp._model_ops import _parse_expression_items
        with pytest.raises(ValueError):
            _parse_expression_items('[{"name": "a"}]', "core_metrics_json")

    def test_evaluate_plain_items(self):
        from comsol_mcp._model_ops import _evaluate_named_expressions

        class FakeModel:
            def evaluate(self, expression):
                if expression == "bad":
                    raise RuntimeError("undefined variable")
                return [[1.0, 2.0]]

        rows = _evaluate_named_expressions(
            FakeModel(),
            [{"name": "a", "expression": "T"}, {"name": "b", "expression": "bad"}],
        )
        assert rows[0]["ok"] is True and rows[0]["last_value"] == 2.0
        assert rows[1]["ok"] is False and "undefined" in rows[1]["error"]


class TestPortIsOpenLoopback:
    def test_listening_loopback_port_is_detected(self):
        import socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        try:
            port = server.getsockname()[1]
            assert _port_is_open("localhost", port) is True
            assert _port_is_open("127.0.0.1", port) is True
        finally:
            server.close()

    def test_closed_loopback_port_returns_quickly(self):
        import time
        start = time.perf_counter()
        assert _port_is_open("localhost", 1) is False
        assert time.perf_counter() - start < 1.0
