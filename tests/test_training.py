"""Unit-тесты для `src/training/{bpr_loss, group_trainer}.py` (Phase 2, шаг 6).

Запуск:
    pytest tests/test_training.py -q
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.group_eval import GroupSample  # noqa: E402
from src.training.bpr_loss import pairwise_bpr_loss  # noqa: E402
from src.training.group_trainer import (  # noqa: E402
    GroupAggregatorTrainer,
    GroupTrainConfig,
    GroupTrainDataset,
    GroupEvalDataset,
    build_user_score_lookup,
    collate_groups,
    compute_pop_counts,
    lookup_per_user_scores,
)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# bpr_loss
# ---------------------------------------------------------------------------

def test_bpr_loss_zero_when_pos_dominates():
    pos = torch.tensor([100.0, 100.0])
    neg = torch.tensor([-100.0, -100.0])
    loss = pairwise_bpr_loss(pos, neg).item()
    assert loss < 1e-6


def test_bpr_loss_symmetry_around_zero():
    pos = torch.tensor([0.0, 0.0])
    neg = torch.tensor([0.0, 0.0])
    loss = pairwise_bpr_loss(pos, neg).item()
    assert _approx(loss, math.log(2.0), tol=1e-6)


def test_bpr_loss_broadcasts_1_to_K():
    pos = torch.tensor([1.0])
    neg = torch.tensor([[0.0, 0.0, 0.0]])  # [1, 3]
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = pairwise_bpr_loss(pos, neg).item()
    assert _approx(loss, expected, tol=1e-6)


def test_bpr_loss_gradient_direction():
    pos = torch.tensor([0.0], requires_grad=True)
    neg = torch.tensor([0.0], requires_grad=True)
    loss = pairwise_bpr_loss(pos, neg)
    loss.backward()
    assert pos.grad.item() < 0      # увеличение pos уменьшает loss
    assert neg.grad.item() > 0      # уменьшение neg уменьшает loss


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

def _toy_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 3, 3, 3],
            "item_idx": [10, 20, 30, 20, 40, 10, 30, 50],
            "score": [5.0, 4.0, 3.0, 6.0, 5.5, 2.0, 1.5, 1.0],
            "rank": [0, 1, 2, 0, 1, 0, 1, 2],
        }
    )


def test_lookup_returns_zero_for_missing_user():
    lookup = build_user_score_lookup(_toy_scores_df())
    out = lookup_per_user_scores(lookup, uid=999, candidates=np.array([10, 20]))
    assert out.dtype == np.float32
    assert np.allclose(out, [0.0, 0.0])


def test_lookup_aligns_scores_with_candidates():
    lookup = build_user_score_lookup(_toy_scores_df())
    # uid=1 has items {10, 20, 30}; candidates {10, 25, 30}
    out = lookup_per_user_scores(lookup, uid=1, candidates=np.array([10, 25, 30]))
    assert _approx(float(out[0]), 5.0)
    assert _approx(float(out[1]), 0.0)  # 25 not in top-K → fill
    assert _approx(float(out[2]), 3.0)


def test_lookup_handles_unsorted_candidates():
    lookup = build_user_score_lookup(_toy_scores_df())
    out = lookup_per_user_scores(lookup, uid=2, candidates=np.array([40, 20]))
    assert _approx(float(out[0]), 5.5)
    assert _approx(float(out[1]), 6.0)


# ---------------------------------------------------------------------------
# pop counts + neg sampling distribution
# ---------------------------------------------------------------------------

def test_compute_pop_counts_zeros_pad_and_sums_to_one():
    df = pd.DataFrame({"item_idx": [1, 1, 1, 2, 2, 3]})
    p = compute_pop_counts(df, n_items=5, smoothing=1.0)
    assert p.shape == (6,)
    assert _approx(float(p[0]), 0.0)
    assert _approx(float(p.sum()), 1.0)
    assert _approx(float(p[1]), 3.0 / 6.0)


def test_train_dataset_excludes_targets_from_negatives():
    # Build a single sample where pop heavily favors a target item; negatives must
    # still come from candidates \ targets only.
    samples = [
        GroupSample(
            members=(1, 2),
            candidates=np.array([10, 20, 30, 40], dtype=np.int64),
            targets=np.array([10, 20], dtype=np.int64),
        )
    ]
    pop = np.zeros(50, dtype=np.float32)
    pop[10] = 0.9        # heavily weighted target
    pop[30] = 0.05
    pop[40] = 0.05
    pop /= pop.sum()

    lookup = build_user_score_lookup(_toy_scores_df())
    ds = GroupTrainDataset(
        samples=samples,
        user_score_lookup=lookup,
        pop_counts=pop,
        n_neg_per_pos=200,
        item_audio=None,
        user_profiles=None,
        uid_to_row=None,
        seed=0,
    )
    item = ds[0]
    neg_items = samples[0].candidates[item["neg_idx"]]
    assert not np.any(np.isin(neg_items, samples[0].targets)), \
        "Negatives must be drawn from candidates \\ targets"
    # All negatives should be in {30, 40} since pop[10]=pop[20]=0 effectively (target-zeroed).
    assert set(neg_items.tolist()).issubset({30, 40})


def test_train_dataset_positive_is_a_target_position():
    samples = [
        GroupSample(
            members=(1, 2),
            candidates=np.array([10, 20, 30], dtype=np.int64),
            targets=np.array([20], dtype=np.int64),
        )
    ]
    pop = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32) * 0 + 0
    pop = np.ones(50, dtype=np.float32) / 49
    pop[0] = 0
    lookup = build_user_score_lookup(_toy_scores_df())
    ds = GroupTrainDataset(samples, lookup, pop_counts=pop, n_neg_per_pos=3, seed=0)
    item = ds[0]
    assert int(item["pos_idx"]) == 1  # 20 is at index 1 in candidates
    assert item["valid"] is True


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def _make_batch_items():
    s_small = GroupSample(
        members=(1, 2),
        candidates=np.array([10, 20], dtype=np.int64),
        targets=np.array([10], dtype=np.int64),
    )
    s_big = GroupSample(
        members=(1, 2, 3),
        candidates=np.array([10, 20, 30, 40], dtype=np.int64),
        targets=np.array([30], dtype=np.int64),
    )
    return [s_small, s_big]


def test_collate_pads_to_batch_max():
    samples = _make_batch_items()
    pop = np.ones(50, dtype=np.float32)
    pop[0] = 0
    pop /= pop.sum()
    lookup = build_user_score_lookup(_toy_scores_df())
    item_audio = np.random.RandomState(0).randn(50, 8).astype(np.float32)
    user_profiles = np.random.RandomState(1).randn(5, 8).astype(np.float32)
    uid_to_row = {1: 0, 2: 1, 3: 2}
    ds = GroupTrainDataset(
        samples=samples,
        user_score_lookup=lookup,
        pop_counts=pop,
        n_neg_per_pos=2,
        item_audio=item_audio,
        user_profiles=user_profiles,
        uid_to_row=uid_to_row,
        seed=0,
    )
    batch = collate_groups([ds[0], ds[1]])
    assert batch["members"].shape == (2, 3)
    assert batch["candidates"].shape == (2, 4)
    assert batch["per_user_scores"].shape == (2, 3, 4)
    assert batch["item_audio"].shape == (2, 4, 8)
    assert batch["user_audio"].shape == (2, 3, 8)
    # group_mask: small group [True, True, False], big group [True, True, True]
    assert batch["group_mask"][0].tolist() == [True, True, False]
    assert batch["group_mask"][1].tolist() == [True, True, True]
    # cand_mask: small [True, True, False, False], big [True, True, True, True]
    assert batch["candidate_mask"][0].tolist() == [True, True, False, False]
    assert batch["candidate_mask"][1].tolist() == [True, True, True, True]
    # pad-positions zeroed
    assert torch.all(batch["per_user_scores"][0, 2, :] == 0)
    assert torch.all(batch["per_user_scores"][0, :, 2:] == 0)


# ---------------------------------------------------------------------------
# toy aggregator + end-to-end fit
# ---------------------------------------------------------------------------

class MeanScoreAggregator(nn.Module):
    """Тривиальный агрегатор: усредняет per_user_scores по реальным членам.

    Параметризован одной обучаемой скалярной шкалой `s`, чтобы оптимизатор
    не упал на пустых params. Для BPR loss скалярная шкала не меняет ranking,
    поэтому val NDCG@K не зависит от обучения — тест проверяет только то,
    что цикл запускается, loss конечен, и evaluate возвращает sane числа.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        group_user_ids,
        candidate_ids,
        per_user_scores,
        audio_embeds_items=None,
        audio_profiles_users=None,
        group_mask=None,
        candidate_mask=None,
    ):
        # mask out padded members so они не вносят 0 в среднее.
        if group_mask is not None:
            gm = group_mask.float().unsqueeze(-1)  # [B, G, 1]
            num = (per_user_scores * gm).sum(dim=1)
            den = gm.sum(dim=1).clamp_min(1.0)
            mean = num / den
        else:
            mean = per_user_scores.mean(dim=1)
        return mean * self.scale  # [B, C_max]


def _make_smoke_setup(tmpdir: Path):
    rng = np.random.default_rng(42)
    n_items = 50
    n_users = 6
    K = 10

    # per-user top-K scores
    rows = []
    for uid in range(1, n_users + 1):
        items = rng.choice(np.arange(1, n_items + 1), size=K, replace=False)
        scores = rng.normal(5.0, 1.0, size=K).astype(np.float32)
        for rk, (it, sc) in enumerate(zip(items, scores)):
            rows.append({"uid": uid, "item_idx": int(it), "score": float(sc), "rank": rk})
    scores_df = pd.DataFrame(rows)
    lookup = build_user_score_lookup(scores_df)

    pop_df = pd.DataFrame({"item_idx": rng.integers(1, n_items + 1, size=500)})
    pop_counts = compute_pop_counts(pop_df, n_items=n_items, smoothing=0.75)

    item_audio = rng.normal(size=(n_items + 1, 8)).astype(np.float32)
    item_audio[0] = 0.0
    user_profiles = rng.normal(size=(n_users, 8)).astype(np.float32)
    uid_to_row = {u: i for i, u in enumerate(range(1, n_users + 1))}

    # Build samples manually with non-empty targets.
    def _make_sample(members, target_subset_n=2):
        cand_set = set()
        for u in members:
            items, _ = lookup[u]
            cand_set.update(items.tolist())
        cands = np.array(sorted(cand_set), dtype=np.int64)
        # pick targets from candidate pool
        targets = rng.choice(cands, size=min(target_subset_n, len(cands)), replace=False)
        targets = np.array(sorted(targets), dtype=np.int64)
        return GroupSample(members=tuple(members), candidates=cands, targets=targets)

    train_samples = [_make_sample([1, 2]), _make_sample([2, 3, 4]), _make_sample([3, 4, 5, 6])]
    val_samples = [_make_sample([1, 3]), _make_sample([2, 5, 6])]

    return {
        "lookup": lookup,
        "pop_counts": pop_counts,
        "item_audio": item_audio,
        "user_profiles": user_profiles,
        "uid_to_row": uid_to_row,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }


def test_trainer_fit_smoke_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="group_trainer_test_"))
    try:
        setup = _make_smoke_setup(tmp)
        cfg = GroupTrainConfig(
            n_epochs=2,
            batch_size=2,
            eval_batch_size=2,
            lr=1e-2,
            n_neg_per_pos=3,
            eval_k=(2, 4),
            early_stop_patience=10,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "agg_run"),
            device="cpu",
        )
        aggregator = MeanScoreAggregator()
        trainer = GroupAggregatorTrainer(
            aggregator=aggregator,
            cfg=cfg,
            user_score_lookup=setup["lookup"],
            pop_counts=setup["pop_counts"],
            item_audio=setup["item_audio"],
            user_profiles=setup["user_profiles"],
            uid_to_row=setup["uid_to_row"],
        )
        result = trainer.fit(setup["train_samples"], setup["val_samples"], verbose=False)
        primary_k = cfg.eval_k[0]
        key = f"best_val_NDCG@{primary_k}"
        assert key in result
        assert math.isfinite(result[key])
        assert 0.0 <= result[key] <= 1.0
        assert Path(result["checkpoint"]).exists()
        assert Path(result["metrics_csv"]).exists()
        assert len(result["history"]) >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_trainer_predict_scores_match_eval_shape():
    tmp = Path(tempfile.mkdtemp(prefix="group_trainer_predict_"))
    try:
        setup = _make_smoke_setup(tmp)
        cfg = GroupTrainConfig(
            n_epochs=1,
            batch_size=2,
            eval_batch_size=2,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "agg_run"),
            device="cpu",
        )
        trainer = GroupAggregatorTrainer(
            aggregator=MeanScoreAggregator(),
            cfg=cfg,
            user_score_lookup=setup["lookup"],
            pop_counts=setup["pop_counts"],
            item_audio=setup["item_audio"],
            user_profiles=setup["user_profiles"],
            uid_to_row=setup["uid_to_row"],
        )
        scores = trainer.predict_group_scores(setup["val_samples"])
        assert len(scores) == len(setup["val_samples"])
        for s, sc in zip(setup["val_samples"], scores):
            assert sc.shape == (s.n_candidates,)
            assert np.all(np.isfinite(sc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


