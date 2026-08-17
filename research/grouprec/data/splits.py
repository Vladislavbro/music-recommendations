"""Global Temporal Split (GTS) for YAMBDA flat interactions.

Pandas port of `references/yambda/benchmarks/yambda/processing/timesplit.py`
(`flat_split_train_val_test`). We keep flat semantics — our DataFrame is
one-row-per-event, not per-user lists.

Segments (default config matches yambda Constants):
    train: timestamp <  test_timestamp - gap - val_size - gap
    val:   timestamp ∈ [test_timestamp - val_size - gap, test_timestamp - gap)
    test:  timestamp >= test_timestamp

Val/test are restricted to users (and optionally items) seen in train.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Mirror yambda Constants (constants.py)
LAST_TIMESTAMP = 26_000_000
TEST_SIZE = 86_400  # 1 day
VAL_SIZE = 86_400  # 1 day
GAP_SIZE = 1_800  # 30 min
TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE  # 25_913_600


@dataclass(frozen=True)
class SplitConfig:
    test_timestamp: int = TEST_TIMESTAMP
    val_size: int = VAL_SIZE
    gap_size: int = GAP_SIZE
    drop_non_train_items: bool = False


def global_temporal_split(
    interactions: pd.DataFrame,
    cfg: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split flat interactions by timestamp into (train, val, test).

    Args:
        interactions: DataFrame with at least columns 'uid', 'item_id',
            'timestamp'. Other columns are preserved.
        cfg: split parameters; `None` → defaults, matching yambda Constants.

    Returns:
        (train, val, test). val/test contain only users present in train.
        If `cfg.val_size == 0`, val is empty (same schema).
    """
    cfg = cfg if cfg is not None else SplitConfig()

    assert cfg.gap_size >= 0
    assert cfg.val_size >= 0

    has_val = cfg.val_size > 0
    train_end_ts = (
        cfg.test_timestamp - cfg.gap_size - cfg.val_size - (cfg.gap_size if has_val else 0)
    )
    assert train_end_ts > 0, (
        f"train_end_ts={train_end_ts} <= 0; check test_timestamp={cfg.test_timestamp}, "
        f"val_size={cfg.val_size}, gap_size={cfg.gap_size}"
    )

    ts = interactions["timestamp"]

    train = interactions.loc[ts < train_end_ts].reset_index(drop=True)
    train_uids = pd.Index(train["uid"].unique())
    train_items = pd.Index(train["item_id"].unique()) if cfg.drop_non_train_items else None

    if has_val:
        val_lo = cfg.test_timestamp - cfg.val_size - cfg.gap_size
        val_hi = cfg.test_timestamp - cfg.gap_size
        val = interactions.loc[(ts >= val_lo) & (ts < val_hi)]
        val = val.loc[val["uid"].isin(train_uids)]
        if train_items is not None:
            val = val.loc[val["item_id"].isin(train_items)]
        val = val.reset_index(drop=True)
    else:
        val = interactions.iloc[0:0].reset_index(drop=True)

    test = interactions.loc[ts >= cfg.test_timestamp]
    test = test.loc[test["uid"].isin(train_uids)]
    if train_items is not None:
        test = test.loc[test["item_id"].isin(train_items)]
    test = test.reset_index(drop=True)

    return train, val, test
