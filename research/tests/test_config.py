"""Тесты слоя конфигов: загрузка YAML, extends, оверрайды, сборка датаклассов."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from grouprec.config import PROJECT_ROOT, build, dump_resolved, load_config
from grouprec.data.group_synthesis import GroupSynthConfig
from grouprec.data.yambda_loader import DataConfig
from grouprec.scorer.inference import InferenceConfig
from grouprec.scorer.train import TrainConfig
from grouprec.training.group_trainer import GroupTrainConfig

CONFIGS = PROJECT_ROOT / "research" / "configs"


@dataclass
class _Toy:
    a: int = 1
    b: str = "x"


# -- extends ----------------------------------------------------------------


def test_extends_merges_parent(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnest:\n  x: 1\n  y: 2\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nnest:\n  y: 9\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"a": 1, "nest": {"x": 1, "y": 9}}
    assert "extends" not in cfg


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("research/configs/nope.yaml")


# -- build ------------------------------------------------------------------


def test_build_applies_section_and_extra():
    toy = build(_Toy, {"a": 5}, b="z")
    assert (toy.a, toy.b) == (5, "z")


def test_build_accepts_none_section():
    assert build(_Toy, None) == _Toy()


def test_build_rejects_unknown_key():
    with pytest.raises(ValueError, match="неизвестные ключи"):
        build(_Toy, {"typo": 1})


# -- dump -------------------------------------------------------------------


def test_dump_resolved_writes_json(tmp_path):
    out = dump_resolved({"train": {"eval_k": (10, 20)}}, tmp_path / "sub" / "config.json")
    assert json.loads(out.read_text(encoding="utf-8")) == {"train": {"eval_k": [10, 20]}}


# -- реальные конфиги ранов -------------------------------------------------


def test_scorer_config_builds_dataclasses():
    cfg = load_config(CONFIGS / "scorer_50m.yaml")
    data_cfg = build(DataConfig, cfg["data"])
    train_cfg = build(TrainConfig, cfg["train"], n_items=1000)
    inf_cfg = build(InferenceConfig, cfg["inference"])

    assert data_cfg.min_popularity == 5
    assert (train_cfg.hidden_dim, train_cfg.n_heads, train_cfg.n_layers) == (64, 2, 2)
    assert train_cfg.n_neg == 1 and train_cfg.mix_uniform == 1.0
    assert inf_cfg.K == 200
    assert inf_cfg.exclude_history is False  # музыка: историю не маскируем


def test_aggregators_config_builds_dataclasses():
    cfg = load_config(CONFIGS / "aggregators_50m.yaml")
    groups_cfg = build(GroupSynthConfig, cfg["groups"])
    train_cfg = build(GroupTrainConfig, cfg["train"], out_dir="artifacts/x")

    assert groups_cfg.size_dist == {2: 0.3, 3: 0.4, 4: 0.2, 5: 0.1}
    assert sum(groups_cfg.size_dist.values()) == pytest.approx(1.0)
    assert list(train_cfg.eval_k) == [10, 20]
    assert train_cfg.fill_score == 0.0
    assert set(cfg["models"]) == {"agree", "groupim", "audio_agree", "group_cross_attn"}


def test_smoke_config_overrides_parent():
    cfg = load_config(CONFIGS / "smoke_local.yaml")
    assert cfg["data"]["smoke"] is True
    assert cfg["train"]["device"] == "cpu"
    assert cfg["train"]["hidden_dim"] == 64  # унаследовано из scorer_50m
    assert cfg["data"]["min_popularity"] == 5  # унаследовано из base


def test_model_params_match_constructor_signatures():
    import inspect

    from grouprec import aggregators

    cfg = load_config(CONFIGS / "aggregators_50m.yaml")
    for name, spec in cfg["models"].items():
        cls = getattr(aggregators, spec["cls"])
        params = set(inspect.signature(cls.__init__).parameters)
        unknown = set(spec["params"]) - params
        assert not unknown, f"{name}: {unknown} нет в {spec['cls']}.__init__"


def test_model_names_are_unique_and_used_by_pairs():
    cfg = load_config(CONFIGS / "aggregators_50m.yaml")
    names = [spec["name"] for spec in cfg["models"].values()]
    assert len(set(names)) == len(names)
    for pair in cfg["eval"]["pairs"]:
        assert set(pair) <= set(names), f"{pair} ссылается на неизвестное имя метода"


def test_model_train_override_is_a_subset_of_train_fields():
    cfg = load_config(CONFIGS / "aggregators_50m.yaml")
    fields = set(build(GroupTrainConfig, cfg["train"]).__dataclass_fields__)
    for name, spec in cfg["models"].items():
        assert set(spec.get("train", {})) <= fields, f"{name}: лишние ключи в train"
    assert cfg["models"]["groupim"]["train"]["reg_loss_weight"] == 0.5


def _parent_of(path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))["extends"]


def test_smoke_configs_never_write_into_real_artifacts():
    """Смок обязан писать в artifacts/smoke — иначе затрёт настоящие чекпоинты."""
    for name in ("smoke_local.yaml", "smoke_aggregators.yaml"):
        cfg = load_config(CONFIGS / name)
        parent = load_config(CONFIGS / _parent_of(CONFIGS / name))
        for key, path in cfg["paths"].items():
            if path != parent["paths"][key]:
                assert path.startswith("artifacts/smoke/"), f"{name}: {key} -> {path}"
        written = {k: v for k, v in cfg["paths"].items() if v.startswith("artifacts/smoke/")}
        assert written, f"{name}: не переопределяет ни одного пути"
