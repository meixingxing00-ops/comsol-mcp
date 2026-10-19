"""Tests for pruning loaded models with fake MPh objects (no COMSOL needed)."""

from pathlib import Path

import comsol_mcp._model as model_mod
import comsol_mcp._server as srv


class FakeJavaModel:
    def __init__(self, tag, label, fail_save=False):
        self._tag = tag
        self._label = label
        self.fail_save = fail_save
        self.saved = []

    def tag(self):
        return self._tag

    def label(self):
        return self._label

    def getFilePath(self):
        return ""

    def save(self, path, copy):
        if self.fail_save:
            raise RuntimeError("disk full")
        Path(path).write_bytes(b"copy")
        self.saved.append((path, copy))


class FakeModel:
    def __init__(self, tag, label, fail_save=False):
        self.java = FakeJavaModel(tag, label, fail_save)


class FakeClient:
    def __init__(self, models):
        self._models = list(models)

    def models(self):
        return list(self._models)

    def remove(self, model):
        self._models.remove(model)


def _setup(monkeypatch, tmp_path, models):
    client = FakeClient(models)
    monkeypatch.setattr(model_mod, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(srv, "_client", client)
    monkeypatch.setattr(srv, "_client_connected", True)
    monkeypatch.setattr(srv, "_current_model", None)
    return client


def test_prune_saves_copy_before_removing(monkeypatch, tmp_path):
    keep = FakeModel("model1", "main.mph")
    other = FakeModel("model2", "Untitled")
    client = _setup(monkeypatch, tmp_path, [keep, other])

    removed, remaining = model_mod._prune_loaded_models_locked(keep)

    assert client.models() == [keep]
    assert [r["tag"] for r in removed] == ["model2"]
    backup = Path(removed[0]["backup_path"])
    assert backup.parent == tmp_path / "outputs" / "pruned"
    assert backup.read_bytes() == b"copy"
    assert other.java.saved[0][1] is True
    assert [r["tag"] for r in remaining] == ["model1"]


def test_prune_keeps_model_when_backup_fails(monkeypatch, tmp_path):
    keep = FakeModel("model1", "main.mph")
    unsaved = FakeModel("model2", "Untitled", fail_save=True)
    client = _setup(monkeypatch, tmp_path, [keep, unsaved])

    removed, remaining = model_mod._prune_loaded_models_locked(keep)

    assert client.models() == [keep, unsaved]
    assert removed == []
    assert sorted(r["tag"] for r in remaining) == ["model1", "model2"]
