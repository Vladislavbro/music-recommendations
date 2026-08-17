"""Group evaluation для Phase 2 (шаг 5).

Сборка test-групп (`GroupSample`) и расчёт NDCG@K по уже посчитанным
group-скорам агрегатора. Сам forward агрегатора живёт в `src/training/`
(шаг 6) и `notebooks/05_eval_groups.ipynb` (шаг 12); здесь — только
data-обвязка и метрики.

Контракты (см. `docs/phase_2_log.md` §«Контракты данных»):
* `candidates = union(top-K каждого члена)`.
* `targets = union(test-listens членов) ∩ candidates` — релевантные items, не
  попавшие в кандидатный пул, исключаются (Phase 2, шаг 5).
* `per_user_scores` агрегатор тащит сам из `scores.parquet`; здесь хранится
  только `candidates`/`targets`/`members`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .metrics import ndcg_from_ranking

GroundTruth = Literal["union", "intersection"]


@dataclass
class GroupSample:
    """Один сэмпл для обучения/оценки агрегатора.

    Attributes:
        members: user_id членов группы, размер `|G|` (2..5).
        candidates: item_idx кандидатного пула `C_G` (union top-K членов),
            int64, без дубликатов.
        targets: item_idx достижимых релевантов (`union(test_listens) ∩ C_G`),
            int64, без дубликатов. Может быть пустым — такие группы по
            умолчанию отфильтровываются `build_group_samples(drop_empty=True)`.
    """

    members: tuple[int, ...]
    candidates: np.ndarray
    targets: np.ndarray

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.targets.shape[0])


def _union(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Union нескольких int-массивов без дубликатов."""
    items = [a for a in arrays if a is not None and len(a) > 0]
    if not items:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(items)).astype(np.int64, copy=False)


def _intersection(arrays: list[np.ndarray]) -> np.ndarray:
    """Intersection нескольких int-массивов. Пустой массив, если хоть один пуст."""
    if not arrays:
        return np.empty(0, dtype=np.int64)
    if any(len(a) == 0 for a in arrays):
        return np.empty(0, dtype=np.int64)
    out = np.asarray(arrays[0], dtype=np.int64)
    for a in arrays[1:]:
        out = np.intersect1d(out, np.asarray(a, dtype=np.int64), assume_unique=False)
        if out.size == 0:
            break
    return out


def build_group_samples(
    groups: list[list[int]],
    user_topk: Mapping[int, np.ndarray],
    user_test_targets: Mapping[int, np.ndarray],
    ground_truth: GroundTruth = "union",
    drop_empty: bool = True,
    drop_missing_member: bool = True,
) -> tuple[list[GroupSample], dict]:
    """Собирает `GroupSample` для каждой группы.

    Args:
        groups: список групп (каждая — список raw user_id).
        user_topk: `{uid: np.ndarray[int64]}` — топ-K item_idx из `scores.parquet`.
            Юзеры без записи трактуются как с пустым top-K.
        user_test_targets: `{uid: np.ndarray[int64]}` — test-listens item_idx
            для каждого uid. Юзеры без записи — без target-вклада.
        ground_truth: «union» (по умолчанию) или «intersection» для агрегации
            таргетов между членами группы.
        drop_empty: фильтровать группы с пустым `targets` (после ∩ candidates).
            Иначе оставляем — `evaluate_aggregator_scores` отдаст 0.0 для них.
        drop_missing_member: фильтровать группы, у которых хотя бы один член
            отсутствует в `user_topk` (нет кэша скоров) — такие нельзя оценить.

    Returns:
        `(samples, stats)`. `stats` — словарь с диагностикой
        (см. `coverage_stats`).
    """
    if ground_truth not in ("union", "intersection"):
        raise ValueError(f"ground_truth must be 'union' or 'intersection', got {ground_truth!r}")

    samples: list[GroupSample] = []
    n_total = len(groups)
    n_missing_member = 0
    n_empty_candidates = 0
    n_empty_targets = 0

    cand_sizes: list[int] = []
    target_sizes: list[int] = []
    size_counts: dict[int, int] = {}

    for members in groups:
        member_tuple = tuple(int(u) for u in members)

        if drop_missing_member and any(u not in user_topk for u in member_tuple):
            n_missing_member += 1
            continue

        # Candidate pool: union top-K.
        topk_arrays = [
            np.asarray(user_topk[u], dtype=np.int64) for u in member_tuple if u in user_topk
        ]
        candidates = _union(topk_arrays)
        if candidates.size == 0:
            n_empty_candidates += 1
            if drop_empty:
                continue

        # Test-listens агрегация.
        tgt_arrays = [
            np.asarray(user_test_targets[u], dtype=np.int64)
            for u in member_tuple
            if u in user_test_targets and len(user_test_targets[u]) > 0
        ]
        if ground_truth == "union":
            raw_targets = _union(tgt_arrays)
        else:
            raw_targets = (
                _intersection(tgt_arrays)
                if len(tgt_arrays) == len(member_tuple)
                else np.empty(0, dtype=np.int64)
            )

        # ∩ candidates: оставляем только достижимые релеванты.
        if raw_targets.size > 0 and candidates.size > 0:
            targets = np.intersect1d(candidates, raw_targets, assume_unique=True)
        else:
            targets = np.empty(0, dtype=np.int64)

        if targets.size == 0:
            n_empty_targets += 1
            if drop_empty:
                continue

        sample = GroupSample(
            members=member_tuple,
            candidates=candidates,
            targets=targets,
        )
        samples.append(sample)
        cand_sizes.append(sample.n_candidates)
        target_sizes.append(sample.n_targets)
        size_counts[sample.size] = size_counts.get(sample.size, 0) + 1

    stats = {
        "n_input_groups": n_total,
        "n_kept": len(samples),
        "n_dropped_missing_member": n_missing_member,
        "n_dropped_empty_candidates": n_empty_candidates,
        "n_dropped_empty_targets": n_empty_targets,
        "candidate_size_mean": float(np.mean(cand_sizes)) if cand_sizes else 0.0,
        "candidate_size_median": float(np.median(cand_sizes)) if cand_sizes else 0.0,
        "target_size_mean": float(np.mean(target_sizes)) if target_sizes else 0.0,
        "target_size_median": float(np.median(target_sizes)) if target_sizes else 0.0,
        "by_size_counts": dict(sorted(size_counts.items())),
        "ground_truth": ground_truth,
    }
    return samples, stats


def evaluate_aggregator_scores(
    samples: list[GroupSample],
    group_scores: list[np.ndarray],
    k_list: list[int] = (10, 20),
) -> dict:
    """NDCG@K на test-группах из уже посчитанных group-скоров агрегатора.

    Args:
        samples: список `GroupSample` от `build_group_samples`.
        group_scores: per-sample `np.ndarray[|C_G|]` скоров агрегатора;
            порядок элементов **совпадает** с `sample.candidates`.
        k_list: K для NDCG (например, `(10, 20)`).

    Returns:
        Словарь:
        ```
        {
            "NDCG@10":          float (mean),
            "NDCG@10_std":      float,
            "NDCG@20":          float,
            "NDCG@20_std":      float,
            "n_samples":        int,
            "by_size": {
                size: {"NDCG@10": ..., "NDCG@20": ..., "n": int},
            },
            "per_sample": {
                "NDCG@10": np.ndarray[N],
                "NDCG@20": np.ndarray[N],
            },
        }
        ```
        `per_sample` отдаём для бутстрапа CI в ноутбуке (шаг 12).
    """
    if len(samples) != len(group_scores):
        raise ValueError(f"len(samples)={len(samples)} != len(group_scores)={len(group_scores)}")
    k_list = list(k_list)
    n = len(samples)

    per_k: dict[int, np.ndarray] = {k: np.zeros(n, dtype=np.float64) for k in k_list}

    for i, (sample, scores) in enumerate(zip(samples, group_scores, strict=True)):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (sample.n_candidates,):
            raise ValueError(
                f"sample #{i}: scores shape {scores.shape} != ({sample.n_candidates},)"
            )
        if sample.n_targets == 0:
            continue  # NDCG = 0 (already zeros)
        order = np.argsort(-scores, kind="stable")
        ranked_items = sample.candidates[order]
        rel_set = set(int(t) for t in sample.targets)
        for k in k_list:
            per_k[k][i] = ndcg_from_ranking(ranked_items, rel_set, k)

    out: dict = {"n_samples": n}
    for k in k_list:
        out[f"NDCG@{k}"] = float(per_k[k].mean()) if n > 0 else 0.0
        out[f"NDCG@{k}_std"] = float(per_k[k].std(ddof=0)) if n > 0 else 0.0
    out["per_sample"] = {f"NDCG@{k}": per_k[k] for k in k_list}

    # Stratify by group size.
    by_size: dict[int, dict] = {}
    sizes = np.array([s.size for s in samples], dtype=np.int64)
    for size in np.unique(sizes):
        idx = np.where(sizes == size)[0]
        entry: dict = {"n": int(idx.size)}
        for k in k_list:
            entry[f"NDCG@{k}"] = float(per_k[k][idx].mean()) if idx.size > 0 else 0.0
        by_size[int(size)] = entry
    out["by_size"] = by_size
    return out


def topk_from_score_cache(scores_df) -> dict[int, np.ndarray]:
    """Хелпер: из `scores.parquet` (см. phase_2_log, шаг 1) собирает `{uid: item_idx[K]}`.

    Ожидаются колонки `uid`, `item_idx`, `score`, `rank`. Сортировка по rank
    не обязательна — для кандидатного пула важен только set.
    """
    required = {"uid", "item_idx"}
    if not required.issubset(scores_df.columns):
        raise ValueError(f"scores_df must have columns {required}, got {set(scores_df.columns)}")
    out: dict[int, np.ndarray] = {}
    for uid, g in scores_df.groupby("uid", sort=False):
        out[int(uid)] = g["item_idx"].to_numpy(dtype=np.int64)
    return out


def test_targets_from_df(test_df) -> dict[int, np.ndarray]:
    """Хелпер: из test-DF (`uid, item_idx, ...`) собирает `{uid: np.ndarray}`."""
    required = {"uid", "item_idx"}
    if not required.issubset(test_df.columns):
        raise ValueError(f"test_df must have columns {required}, got {set(test_df.columns)}")
    out: dict[int, np.ndarray] = {}
    for uid, g in test_df.groupby("uid", sort=False):
        out[int(uid)] = g["item_idx"].to_numpy(dtype=np.int64)
    return out
