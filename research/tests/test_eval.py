"""Unit-тесты для `research/grouprec/eval/{metrics, group_eval}.py` (Phase 2, шаг 5).

Запуск:
    pytest research/tests/test_eval.py -q
"""

from __future__ import annotations

import math

import numpy as np
from grouprec.eval.group_eval import (
    GroupSample,
    build_group_samples,
    evaluate_aggregator_scores,
)
from grouprec.eval.metrics import (
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k,
    ndcg_from_ranking,
    ranking_ndcg_at_k,
)


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_dcg_at_k_manual():
    # rel = [1, 0, 1, 0] @ k=4 -> 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    rel = np.array([[1, 0, 1, 0]])
    out = dcg_at_k(rel, k=4)
    assert _approx(out[0], 1.0 + 0.5), out


def test_idcg_at_k_manual():
    # n_relevant=3, k=5 -> ideal = [1,1,1,0,0] -> 1 + 1/log2(3) + 1/log2(4)
    expected = 1.0 + 1.0 / math.log2(3.0) + 0.5
    out = idcg_at_k(np.array([3]), k=5)
    assert _approx(out[0], expected), out


def test_ndcg_perfect_ranking():
    # 3 релеванта в первых 3 позициях, k=5 -> NDCG = 1.0
    rel = np.array([[1, 1, 1, 0, 0]])
    n_rel = np.array([3])
    assert _approx(ndcg_at_k(rel, n_rel, k=5)[0], 1.0)


def test_ndcg_zero_relevants():
    # IDCG = 0 -> NDCG = 0 (safe-divide)
    rel = np.array([[0, 0, 0, 0, 0]])
    n_rel = np.array([0])
    assert _approx(ndcg_at_k(rel, n_rel, k=5)[0], 0.0)


def test_ndcg_truncates_at_k():
    # rel = [0, 1, 0, 1], n_rel = 2, k=2 -> dcg = 1/log2(3) , idcg = 1 + 1/log2(3)
    rel = np.array([[0, 1, 0, 1]])
    n_rel = np.array([2])
    dcg = 1.0 / math.log2(3.0)
    idcg = 1.0 + 1.0 / math.log2(3.0)
    out = ndcg_at_k(rel, n_rel, k=2)
    assert _approx(out[0], dcg / idcg), out


def test_ranking_ndcg_uses_score_order():
    # candidates: 4 items, scores [3.0, 1.0, 2.0, 0.5], targets_mask = [1, 0, 1, 0]
    # sorted by score desc: items[0]=relevant, items[2]=relevant, ...
    # rel sorted: [1, 1, 0, 0]; n_rel=2; k=2 -> NDCG=1.0
    scores = np.array([[3.0, 1.0, 2.0, 0.5]])
    mask = np.array([[1, 0, 1, 0]])
    out = ranking_ndcg_at_k(scores, mask, [2, 4])
    assert _approx(out[2][0], 1.0)
    assert _approx(out[4][0], 1.0)


def test_ranking_ndcg_imperfect():
    # scores [3, 2, 1, 0], targets at items 2 and 3 (lowest scores) -> NDCG@2 = 0
    scores = np.array([[3.0, 2.0, 1.0, 0.0]])
    mask = np.array([[0, 0, 1, 1]])
    out = ranking_ndcg_at_k(scores, mask, [2, 4])
    # k=2: top-2 are items 0,1; both irrelevant -> NDCG@2 = 0
    assert _approx(out[2][0], 0.0)
    # k=4: full ranking, rel = [0,0,1,1]; dcg = 1/log2(4)+1/log2(5)
    dcg = 1.0 / math.log2(4.0) + 1.0 / math.log2(5.0)
    idcg = 1.0 + 1.0 / math.log2(3.0)  # 2 relevant
    assert _approx(out[4][0], dcg / idcg, tol=1e-9), out


def test_ndcg_from_ranking_matches_low_level():
    ranked = [10, 20, 30, 40, 50]
    relevant = {20, 50}
    expected = (1.0 / math.log2(3.0) + 1.0 / math.log2(6.0)) / (1.0 + 1.0 / math.log2(3.0))
    assert _approx(ndcg_from_ranking(ranked, relevant, k=5), expected)
    # k=1: top-1=10, not in relevant -> 0
    assert _approx(ndcg_from_ranking(ranked, relevant, k=1), 0.0)


def test_ndcg_batch_multiple_rows():
    # Row 1: perfect; Row 2: zero; Row 3: imperfect (matches test_ranking_ndcg_imperfect)
    scores = np.array(
        [
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    mask = np.array(
        [
            [1, 1, 0, 0],  # perfect @2,4
            [0, 0, 0, 0],  # no relevants
            [0, 0, 1, 1],  # bottom @2 -> 0
        ]
    )
    out = ranking_ndcg_at_k(scores, mask, [2])
    assert _approx(out[2][0], 1.0)
    assert _approx(out[2][1], 0.0)
    assert _approx(out[2][2], 0.0)


# ---------------------------------------------------------------------------
# group_eval.py
# ---------------------------------------------------------------------------


def test_build_group_samples_union_basic():
    # 2 users, top-K {u=1: [10,20,30], u=2: [20,30,40]}, test {u=1:[20,50], u=2:[40,60]}
    user_topk = {1: np.array([10, 20, 30]), 2: np.array([20, 30, 40])}
    test_tgt = {1: np.array([20, 50]), 2: np.array([40, 60])}
    samples, stats = build_group_samples([[1, 2]], user_topk, test_tgt, ground_truth="union")
    assert len(samples) == 1
    s = samples[0]
    # candidates = union = {10,20,30,40}
    assert set(s.candidates.tolist()) == {10, 20, 30, 40}
    # raw targets union = {20,50,40,60}; ∩ candidates = {20, 40}
    assert set(s.targets.tolist()) == {20, 40}
    assert stats["n_kept"] == 1
    assert stats["n_dropped_empty_targets"] == 0


def test_build_group_samples_intersection():
    user_topk = {1: np.array([10, 20, 30]), 2: np.array([20, 30, 40])}
    test_tgt = {1: np.array([20, 30, 50]), 2: np.array([30, 40])}
    samples, _ = build_group_samples([[1, 2]], user_topk, test_tgt, ground_truth="intersection")
    assert len(samples) == 1
    # intersection raw = {30}; ∩ candidates ({10,20,30,40}) = {30}
    assert set(samples[0].targets.tolist()) == {30}


def test_build_group_samples_drops_empty_target():
    user_topk = {1: np.array([10, 20]), 2: np.array([20, 30])}
    # No member has test listens in the candidate pool
    test_tgt = {1: np.array([99]), 2: np.array([100])}
    samples, stats = build_group_samples([[1, 2]], user_topk, test_tgt, drop_empty=True)
    assert samples == []
    assert stats["n_dropped_empty_targets"] == 1
    # And without drop:
    samples_keep, stats_keep = build_group_samples([[1, 2]], user_topk, test_tgt, drop_empty=False)
    assert len(samples_keep) == 1
    assert samples_keep[0].n_targets == 0
    assert stats_keep["n_dropped_empty_targets"] == 1


def test_build_group_samples_drops_missing_member():
    user_topk = {1: np.array([10, 20])}
    test_tgt = {1: np.array([10]), 2: np.array([10])}
    samples, stats = build_group_samples([[1, 2]], user_topk, test_tgt, drop_missing_member=True)
    assert samples == []
    assert stats["n_dropped_missing_member"] == 1


def test_build_group_samples_invalid_ground_truth():
    try:
        build_group_samples(
            [[1, 2]], {1: np.array([10]), 2: np.array([20])}, {}, ground_truth="weird"
        )  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid ground_truth")


def test_evaluate_aggregator_perfect_ranking():
    # Group: candidates [10, 20, 30, 40], targets {10, 30}
    # Aggregator gives top scores to targets -> NDCG@K = 1.0 for any K >= 2
    sample = GroupSample(
        members=(1, 2),
        candidates=np.array([10, 20, 30, 40], dtype=np.int64),
        targets=np.array([10, 30], dtype=np.int64),
    )
    scores = np.array([5.0, 1.0, 4.0, 0.5])  # ranks targets first
    out = evaluate_aggregator_scores([sample], [scores], k_list=[2, 4])
    assert _approx(out["NDCG@2"], 1.0)
    assert _approx(out["NDCG@4"], 1.0)
    assert out["n_samples"] == 1


def test_evaluate_aggregator_size_breakdown():
    s2 = GroupSample(
        members=(1, 2),
        candidates=np.array([10, 20], dtype=np.int64),
        targets=np.array([10], dtype=np.int64),
    )
    s3 = GroupSample(
        members=(1, 2, 3),
        candidates=np.array([10, 20], dtype=np.int64),
        targets=np.array([20], dtype=np.int64),
    )
    # s2: target at rank 0 -> NDCG = 1.0; s3: target at rank 1 -> NDCG = 1/log2(3)
    out = evaluate_aggregator_scores(
        [s2, s3],
        [np.array([2.0, 1.0]), np.array([2.0, 1.0])],
        k_list=[2],
    )
    assert "by_size" in out
    assert 2 in out["by_size"] and 3 in out["by_size"]
    assert _approx(out["by_size"][2]["NDCG@2"], 1.0)
    assert _approx(out["by_size"][3]["NDCG@2"], 1.0 / math.log2(3.0))


def test_evaluate_aggregator_score_shape_mismatch():
    sample = GroupSample(
        members=(1, 2),
        candidates=np.array([10, 20, 30], dtype=np.int64),
        targets=np.array([10], dtype=np.int64),
    )
    try:
        evaluate_aggregator_scores([sample], [np.array([1.0, 2.0])], k_list=[2])
    except ValueError:
        return
    raise AssertionError("expected ValueError on score shape mismatch")


def test_evaluate_aggregator_empty_targets_returns_zero():
    # If we keep groups with empty targets (drop_empty=False), they should contribute 0.0
    sample = GroupSample(
        members=(1, 2),
        candidates=np.array([10, 20], dtype=np.int64),
        targets=np.array([], dtype=np.int64),
    )
    out = evaluate_aggregator_scores([sample], [np.array([1.0, 2.0])], k_list=[2])
    assert _approx(out["NDCG@2"], 0.0)
