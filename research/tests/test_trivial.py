"""Тесты необучаемых агрегаторов AVG / LM / MP."""

from __future__ import annotations

import numpy as np
from grouprec.eval.group_eval import GroupSample
from grouprec.eval.trivial import TRIVIAL_AGGREGATORS, trivial_group_scores

CANDIDATES = np.array([10, 20, 30], dtype=np.int64)
# uid 1 знает все три item'а, uid 2 — только 20; остальное добирается fill.
LOOKUP = {
    1: (np.array([10, 20, 30], dtype=np.int64), np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    2: (np.array([20], dtype=np.int64), np.array([9.0], dtype=np.float32)),
}
SAMPLE = GroupSample(members=(1, 2), candidates=CANDIDATES, targets=np.array([20], dtype=np.int64))


def _scores(name: str, fill: float = 0.0) -> np.ndarray:
    return trivial_group_scores([SAMPLE], LOOKUP, TRIVIAL_AGGREGATORS[name], fill=fill)[0]


def test_avg_averages_over_members():
    np.testing.assert_allclose(_scores("AVG"), [0.5, 5.5, 1.5])


def test_least_misery_takes_min():
    np.testing.assert_allclose(_scores("LM"), [0.0, 2.0, 0.0])


def test_max_pleasure_takes_max():
    np.testing.assert_allclose(_scores("MP"), [1.0, 9.0, 3.0])


def test_fill_applies_to_items_outside_topk():
    # item 10 и 30 отсутствуют у uid 2 → в LM берётся fill, а не значение uid 1.
    np.testing.assert_allclose(_scores("LM", fill=-5.0), [-5.0, 2.0, -5.0])


def test_output_shape_and_dtype():
    scores = _scores("AVG")
    assert scores.shape == (SAMPLE.n_candidates,)
    assert scores.dtype == np.float32


def test_unknown_user_falls_back_to_fill():
    sample = GroupSample(members=(1, 999), candidates=CANDIDATES, targets=CANDIDATES[:1])
    scores = trivial_group_scores([sample], LOOKUP, TRIVIAL_AGGREGATORS["MP"])[0]
    np.testing.assert_allclose(scores, [1.0, 2.0, 3.0])
