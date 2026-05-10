"""YAMBDA dataset loader (flat-multievent flavor, pandas).

Phase 1 scope: only interactions. Audio embeddings (14GB) are deferred to
Phase 2 — see CLAUDE.md и docs/PHASE_1_LOG.md.

Columns of the loaded interactions DataFrame (после `to_pandas()`):
    uid:                  uint32
    item_id:              uint32
    timestamp:            uint32 (seconds, anonymized)
    is_organic:           uint8
    played_ratio_pct:     float (NaN for non-listen events)
    track_length_seconds: float (NaN for non-listen events)
    event_type:           category (listen/like/unlike/dislike/undislike)

YAMBDA Constants used:
    TRACK_LISTEN_THRESHOLD = 50    # ≥50% played counts as a real listen
    TEST_TIMESTAMP         = 25_913_600  (LAST_TIMESTAMP - 1 day)
    LAST_TIMESTAMP         = 26_000_000
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

TRACK_LISTEN_THRESHOLD = 50  # mirrors yambda Constants.TRACK_LISTEN_THRESHOLD
DEFAULT_MIN_POPULARITY = 5


def load_yambda(
    flavor: Literal["50m"] = "50m",
    cache_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load YAMBDA flat-multievent flavor through HuggingFace `datasets`.

    Audio embeddings and meta tables are NOT loaded in Phase 1 — only
    `interactions` is returned. Returned dict shape is kept ready for Phase 2.

    Args:
        flavor: only "50m" is supported in Phase 1.
        cache_dir: optional HF cache dir (e.g. '/content/hf_cache' for Colab).

    Returns:
        {"interactions": <DataFrame with the columns above>}
    """
    if flavor != "50m":
        raise NotImplementedError(
            f"Phase 1 only supports flavor='50m', got {flavor!r}. "
            "Larger flavors are deferred to Phase 2."
        )

    from datasets import load_dataset  # local import — heavy dep

    ds = load_dataset(
        "yandex/yambda",
        "flat-multievent-50m",
        split="train",
        cache_dir=cache_dir,
    )
    interactions = ds.to_pandas()
    return {"interactions": interactions}


def filter_listens(
    df: pd.DataFrame,
    threshold_pct: int = TRACK_LISTEN_THRESHOLD,
) -> pd.DataFrame:
    """Keep only `event_type == 'listen' AND played_ratio_pct >= threshold_pct`.

    Returns a copy with index reset and columns `uid, item_id, timestamp`.
    """
    is_listen = df["event_type"] == "listen"
    is_engaged = df["played_ratio_pct"] >= threshold_pct
    out = df.loc[is_listen & is_engaged, ["uid", "item_id", "timestamp"]]
    return out.reset_index(drop=True)


def filter_min_popularity(
    df: pd.DataFrame,
    min_count: int = DEFAULT_MIN_POPULARITY,
) -> pd.DataFrame:
    """Drop interactions with items that occur fewer than `min_count` times.

    Phase-1 decision (PHASE_1_LOG.md): cuts long-tail catalog noise that
    embeddings cannot learn anyway. Apply BEFORE building item_id_to_idx
    so that the remap covers exactly the kept items.
    """
    counts = df["item_id"].value_counts()
    keep = counts.index[counts >= min_count]
    out = df[df["item_id"].isin(keep)]
    return out.reset_index(drop=True)


def build_item_id_to_idx(df: pd.DataFrame) -> dict[int, int]:
    """Map raw item_id -> contiguous indices 1..N (0 reserved for padding).

    Mirrors `references/yambda/benchmarks/models/sasrec/data.py:72`.
    """
    unique = sorted(df["item_id"].unique().tolist())
    return {int(item_id): i + 1 for i, item_id in enumerate(unique)}


def apply_item_remap(df: pd.DataFrame, item_id_to_idx: dict[int, int]) -> pd.DataFrame:
    """Replace raw item_id column with contiguous index `item_idx` (1..N)."""
    out = df.copy()
    out["item_idx"] = out["item_id"].map(item_id_to_idx).astype("int64")
    return out


def subsample_users(
    df: pd.DataFrame,
    n_users: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Keep all interactions of a random subset of `n_users` users.

    Used for local smoke tests on M4 Pro (full 50m has ~9.2k users; smoke at 5k).
    """
    rng = pd.Series(df["uid"].unique()).sample(
        n=min(n_users, df["uid"].nunique()), random_state=seed
    )
    out = df[df["uid"].isin(rng.values)]
    return out.reset_index(drop=True)
