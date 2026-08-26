"""Общая сборка входов для групповых ранов: данные, кэш скоров, аудио, группы.

Обучение (`scripts/train_aggregators.py`) и оценка (`scripts/eval_groups.py`)
поднимают один и тот же контекст — иначе split или item-ремап разъедутся между
шагами и числа перестанут быть сравнимыми.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, build
from .data.splits import SplitConfig, global_temporal_split
from .data.yambda_loader import DataConfig, apply_item_remap, prepare_interactions
from .eval.group_eval import topk_from_score_cache
from .training.group_trainer import UserScoreLookup, build_user_score_lookup

__all__ = ["GroupContext", "load_group_context", "resolve_paths"]


@dataclass
class GroupContext:
    """Всё, что нужно и трейнеру, и evaluator'у, поднятое один раз."""

    n_items: int
    user_pool: list[int]
    user_topk: dict[int, np.ndarray]
    user_score_lookup: UserScoreLookup
    item_audio: np.ndarray
    user_profiles: np.ndarray
    uid_to_row: dict[int, int]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    paths: dict[str, Path]


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {name: PROJECT_ROOT / value for name, value in cfg["paths"].items()}


def load_group_context(cfg: dict[str, Any]) -> GroupContext:
    paths = resolve_paths(cfg)
    item_id_to_idx = _read_pickle(paths["scorer"] / "item_id_to_idx.pkl")

    df = prepare_interactions(build(DataConfig, cfg.get("data")))
    df = apply_item_remap(df, item_id_to_idx)
    train_df, val_df, test_df = global_temporal_split(df, build(SplitConfig, cfg.get("split")))

    scores_df = pd.read_parquet(paths["scores_cache"])
    user_topk = topk_from_score_cache(scores_df)
    item_audio, user_profiles, uid_to_row = _load_audio(paths["audio"])

    return GroupContext(
        n_items=max(item_id_to_idx.values()),
        user_pool=sorted(user_topk.keys()),
        user_topk=user_topk,
        user_score_lookup=build_user_score_lookup(scores_df),
        item_audio=item_audio,
        user_profiles=user_profiles,
        uid_to_row=uid_to_row,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        paths=paths,
    )


def _load_audio(audio_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    return (
        np.load(audio_dir / "embeddings.npy"),
        np.load(audio_dir / "user_profiles.npy"),
        _read_pickle(audio_dir / "uid_to_row.pkl"),
    )


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)
