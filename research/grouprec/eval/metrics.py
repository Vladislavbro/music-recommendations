"""NDCG@K — векторизованные реализации для Phase 2 (шаг 5).

Два уровня API:

* `ndcg_at_k(relevances, n_relevant, k)` — низкоуровневый: принимает уже
  отсортированную по predicted score (desc) бинарную матрицу `[B, L]`
  релевантностей и `n_relevant[B]` (число релевантных, достижимых в ranking;
  IDCG = sum(1/log2(i+2)) для i ∈ [0, min(k, n_relevant)-1]).
* `ranking_ndcg_at_k(scores, targets_mask, k_list)` — высокоуровневый:
  сортирует scores per-row и вызывает первый.

Контракт Phase 2: relevant items, не попавшие в `C_G`, в `targets_mask` не
заходят (см. phase_2_log.md, шаг 5, «Out-of-pool»). Таким образом IDCG
нормирует только по достижимым релевантам.
"""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> np.ndarray:
    """DCG@K. `relevances` — `[B, L]` бинарные, отсортированы по predicted score desc."""
    if relevances.ndim != 2:
        raise ValueError(f"relevances must be 2D, got shape {relevances.shape}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    cut = relevances[:, :k].astype(np.float64, copy=False)
    discounts = 1.0 / np.log2(np.arange(cut.shape[1]) + 2.0)
    return (cut * discounts).sum(axis=1)


def idcg_at_k(n_relevant: np.ndarray, k: int) -> np.ndarray:
    """IDCG@K для `n_relevant[B]` достижимых релевантов. `[B]` float64."""
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    n_relevant = np.asarray(n_relevant, dtype=np.int64)
    ideal = np.minimum(n_relevant, k)
    cum = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(k) + 2.0))])
    # cum[m] = IDCG для m достижимых релевантов (m ∈ [0, k]).
    return cum[ideal]


def ndcg_at_k(
    relevances: np.ndarray,
    n_relevant: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """NDCG@K из `[B, L]` бинарных релевантностей (sorted desc) и `n_relevant[B]`.

    Возвращает `[B]` float64. Группы с `n_relevant == 0` получают 0.0.
    """
    dcg = dcg_at_k(relevances, k)
    idcg = idcg_at_k(n_relevant, k)
    out = np.zeros_like(dcg)
    nz = idcg > 0
    out[nz] = dcg[nz] / idcg[nz]
    return out


def ranking_ndcg_at_k(
    scores: np.ndarray,
    targets_mask: np.ndarray,
    k_list: list[int],
) -> dict[int, np.ndarray]:
    """NDCG@K на нескольких K из плотных `scores[B, L]` и бинарной маски таргетов.

    Args:
        scores: предсказанные скоры, выше — лучше.
        targets_mask: 1, если кандидат — релевант; 0 иначе. Не-кандидаты должны
            быть исключены из обеих матриц до вызова (или маска уже = 0 для них
            и `scores = -inf`).
        k_list: значения K (например, [10, 20]).

    Returns:
        dict `{k: ndarray[B]}`.
    """
    if scores.shape != targets_mask.shape:
        raise ValueError(
            f"shape mismatch: scores {scores.shape} vs targets_mask {targets_mask.shape}"
        )
    # Argsort по -scores — устойчивый порядок не критичен (бинарные релевантности).
    order = np.argsort(-scores, axis=1)
    sorted_rel = np.take_along_axis(targets_mask, order, axis=1)
    n_rel = targets_mask.sum(axis=1)
    k_max = max(k_list)
    sorted_rel = sorted_rel[:, :k_max]  # экономия на больших |C_G|
    return {k: ndcg_at_k(sorted_rel, n_rel, k) for k in k_list}


def ndcg_from_ranking(
    ranked_items: np.ndarray | list[int],
    relevant_set: set[int],
    k: int = 10,
) -> float:
    """Удобная single-query NDCG@K из уже отранжированного списка `ranked_items`.

    `ranked_items` — id-шники, отсортированные по predicted score desc;
    `relevant_set` — множество достижимых релевантов (после ∩ candidates).
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    items = np.asarray(list(ranked_items)[:k], dtype=np.int64)
    rel = np.array([int(i) in relevant_set for i in items], dtype=np.float64)
    n_rel = len(relevant_set)
    dcg = (rel / np.log2(np.arange(len(rel)) + 2.0)).sum()
    ideal_n = min(n_rel, k)
    if ideal_n == 0:
        return 0.0
    idcg = (1.0 / np.log2(np.arange(ideal_n) + 2.0)).sum()
    return float(dcg / idcg)
