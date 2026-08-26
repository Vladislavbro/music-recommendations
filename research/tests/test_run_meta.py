"""Тесты `run.json`: состав полей и сериализация."""

from __future__ import annotations

import json
from dataclasses import dataclass

from grouprec.utils.run_meta import write_run_json


@dataclass
class _Cfg:
    lr: float = 1e-3
    device: str = "cpu"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_writes_expected_sections(tmp_path):
    payload = _read(write_run_json(tmp_path, config=_Cfg(), metrics={"NDCG@10": 0.09}))
    assert set(payload) == {"created_at", "git", "env", "config", "metrics"}
    assert payload["metrics"]["NDCG@10"] == 0.09
    assert payload["env"]["python"]


def test_dataclass_config_is_expanded(tmp_path):
    payload = _read(write_run_json(tmp_path, config=_Cfg(lr=0.5), metrics={}))
    assert payload["config"] == {"lr": 0.5, "device": "cpu"}


def test_dict_config_passes_through(tmp_path):
    payload = _read(write_run_json(tmp_path, config={"n_boot": 100}, metrics={}))
    assert payload["config"] == {"n_boot": 100}


def test_creates_missing_directory(tmp_path):
    out = write_run_json(tmp_path / "deep" / "nested", config={}, metrics={})
    assert out.exists() and out.name == "run.json"


def test_git_section_reports_revision_and_dirty(tmp_path):
    git = _read(write_run_json(tmp_path, config={}, metrics={}))["git"]
    assert set(git) == {"revision", "dirty"}
    assert git["revision"] is None or len(git["revision"]) == 40


def test_non_serializable_values_do_not_crash(tmp_path):
    payload = _read(write_run_json(tmp_path, config={}, metrics={"path": tmp_path}))
    assert payload["metrics"]["path"] == str(tmp_path)
