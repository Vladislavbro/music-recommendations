"""Синтез random-групп для Phase 2 (шаг 3).

Эфемерные группы: для каждой группы выбираем размер из заданного
распределения, затем `size` уникальных пользователей из `user_pool`
(without replacement внутри группы, with replacement между группами).
Стандартный setup AGREE / GroupIM для ad-hoc групп.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SIZE_DIST = {2: 0.3, 3: 0.4, 4: 0.2, 5: 0.1}


@dataclass
class GroupSynthConfig:
    """Параметры синтеза random-групп: сколько, каких размеров, с каким seed."""

    seed: int = 42
    n_train: int = 10_000
    n_val: int = 2_000
    n_test: int = 2_000
    size_dist: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_SIZE_DIST))

    def __post_init__(self) -> None:
        # YAML отдаёт ключи маппинга строками, если они записаны в кавычках.
        self.size_dist = {int(k): float(v) for k, v in self.size_dist.items()}


def synthesize_group_splits(
    user_pool: list[int],
    cfg: GroupSynthConfig | None = None,
) -> dict[str, list[list[int]]]:
    """Синтезирует train/val/test группы одним seed-деревом.

    Один `cfg.seed` → один и тот же split групп для всех методов, чтобы
    сравнение агрегаторов было честным.

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    cfg = cfg if cfg is not None else GroupSynthConfig()
    rng = np.random.default_rng(cfg.seed)
    counts = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}
    return {
        split: synthesize_random_groups(
            user_pool, n, cfg.size_dist, seed=int(rng.integers(1 << 30))
        )
        for split, n in counts.items()
    }


def synthesize_random_groups(
    user_pool: list[int],
    n_groups: int,
    size_distribution: dict[int, float],
    seed: int = 0,
) -> list[list[int]]:
    """Сэмплирует `n_groups` случайных групп из `user_pool`.

    Args:
        user_pool: список user_id, из которого набираем группы.
        n_groups: сколько групп вернуть.
        size_distribution: dict `{size: probability}`, вероятности должны
            суммироваться в 1.0 (с точностью до 1e-6). Например
            `{2: 0.3, 3: 0.4, 4: 0.2, 5: 0.1}`.
        seed: для воспроизводимости.

    Returns:
        Список из `n_groups` списков user_id. Внутри каждой группы id
        уникальны; между группами повторы разрешены.
    """
    assert n_groups > 0, f"n_groups={n_groups} must be > 0"
    assert len(size_distribution) > 0, "size_distribution is empty"

    sizes = np.array(sorted(size_distribution.keys()), dtype=np.int64)
    probs = np.array([size_distribution[int(s)] for s in sizes], dtype=np.float64)
    assert (sizes >= 2).all(), f"all sizes must be >= 2, got {sizes.tolist()}"
    assert (probs >= 0).all(), "probabilities must be non-negative"
    assert abs(probs.sum() - 1.0) < 1e-6, (
        f"size_distribution probabilities must sum to 1.0, got {probs.sum():.6f}"
    )

    max_size = int(sizes.max())
    assert len(user_pool) >= max_size, (
        f"user_pool size {len(user_pool)} < max group size {max_size}"
    )

    rng = np.random.default_rng(seed)
    pool = np.asarray(user_pool, dtype=np.int64)

    group_sizes = rng.choice(sizes, size=n_groups, p=probs)
    groups: list[list[int]] = []
    for size in group_sizes:
        members = rng.choice(pool, size=int(size), replace=False)
        groups.append([int(x) for x in members])
    return groups


if __name__ == "__main__":
    # Smoke
    pool = list(range(100))
    dist = {2: 0.3, 3: 0.4, 4: 0.2, 5: 0.1}
    groups = synthesize_random_groups(pool, n_groups=10_000, size_distribution=dist, seed=0)
    assert len(groups) == 10_000
    sizes = np.array([len(g) for g in groups])
    print(
        f"n_groups={len(groups)}, size mean={sizes.mean():.3f}, "
        f"counts={dict(zip(*np.unique(sizes, return_counts=True), strict=True))}"
    )
    # Каждая группа — уникальные участники
    assert all(len(set(g)) == len(g) for g in groups)
    # Детерминизм по seed
    g2 = synthesize_random_groups(pool, 10_000, dist, seed=0)
    assert groups == g2
    g3 = synthesize_random_groups(pool, 10_000, dist, seed=1)
    assert groups != g3
    print("smoke OK")
