"""Загрузка YAML-конфигов ранов (`research/configs/*.yaml`) и сборка датаклассов.

cfg = load_config("research/configs/scorer_50m.yaml")
train_cfg = build(TrainConfig, cfg["train"], n_items=n_items)
dump_resolved(cfg, out_dir / "config.resolved.json")
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["PROJECT_ROOT", "build", "dump_resolved", "load_config"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_with_extends(PROJECT_ROOT / path)  # абсолютный `path` PROJECT_ROOT не тронет


def _load_with_extends(cfg_path: Path) -> dict[str, Any]:
    """Разворачивает цепочку `extends`: родитель, поверх него сам файл.

    Путь родителя — относительно файла с `extends`, а не корня репо, поэтому
    рекурсия идёт здесь, а не через резолв пути в `load_config`.
    """
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"load_config: {cfg_path} должен содержать mapping, получен {type(raw)}")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_load_with_extends(cfg_path.parent / str(parent)), raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Сливает вложенные секции; `override` выигрывает.

    Обычный `update` затёр бы секцию целиком: `train: {lr: 1e-4}` в дочернем
    конфиге снёс бы все остальные ключи `train` из родителя.
    """
    out = dict(base)  # копия: входы не мутируем
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build[T](cls: type[T], section: dict[str, Any] | None, **extra: Any) -> T:
    """Датакласс из секции YAML плюс `**extra` — значения, известные только в рантайме (`n_items`).

    Не `TrainConfig(**cfg["train"])`, потому что опечатка в YAML должна ронять ран
    с внятным сообщением, а не уводить его на дефолты.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"build: {cls!r} не датакласс")

    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {**(section or {}), **extra}

    unknown = sorted(set(kwargs) - field_names)
    if unknown:
        raise ValueError(
            f"build({cls.__name__}): неизвестные ключи {unknown}; доступны: {sorted(field_names)}"
        )
    return cls(**kwargs)  # type: ignore[return-value]


def dump_resolved(cfg: dict[str, Any], path: str | Path) -> Path:
    """Пишет итоговый конфиг рядом с артефактами рана — чтобы ран был воспроизводим."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple | set):
        return list(obj)
    raise TypeError(f"не сериализуется в json: {type(obj)}")
