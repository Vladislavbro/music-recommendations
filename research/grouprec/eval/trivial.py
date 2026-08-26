"""Необучаемые агрегаторы: AVG, LM (Least Misery), MP (Max Pleasure).

Работают поверх того же кэша скоров, что и обучаемые: item вне top-K члена
получает `fill`, поэтому LM систематически проигрывает — это свойство усечения,
а не метода (см. logs/phase_2_log.md).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..training.group_trainer import UserScoreLookup, lookup_per_user_scores
from .group_eval import GroupSample

__all__ = ["TRIVIAL_AGGREGATORS", "trivial_group_scores"]

TRIVIAL_AGGREGATORS: dict[str, Callable[..., np.ndarray]] = {
    "AVG": np.mean,
    "LM": np.min,
    "MP": np.max,
}


def trivial_group_scores(
    samples: list[GroupSample],
    score_lookup: UserScoreLookup,
    agg_fn: Callable[..., np.ndarray],
    fill: float = 0.0,
) -> list[np.ndarray]:
    """Групповые скоры без обучения: `agg_fn` по оси членов группы."""
    out = []
    for sample in samples:
        per_user = np.stack(
            [
                lookup_per_user_scores(score_lookup, int(u), sample.candidates, fill=fill)
                for u in sample.members
            ],
            axis=0,
        )
        out.append(agg_fn(per_user, axis=0).astype(np.float32))
    return out
