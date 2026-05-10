"""Filesystem cache helpers for artifacts (pickle + parquet round-trips)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_pickle(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    return path


def load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)
