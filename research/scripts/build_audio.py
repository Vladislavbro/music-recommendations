"""Аудиоартефакты: подмножество эмбеддингов каталога и профили пользователей.

    uv run python research/scripts/build_audio.py --config research/configs/aggregators_50m.yaml
    uv run python research/scripts/build_audio.py --config ... --profiles-only

`--profiles-only` пропускает выкачивание `embeddings.parquet` с HF (13.8 ГБ) и
переиспользует уже собранный `audio/embeddings.npy`.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from grouprec.config import build, dump_resolved, load_config
from grouprec.data.audio_embeddings import build_user_audio_profiles, extract_audio_subset
from grouprec.data.splits import SplitConfig, global_temporal_split
from grouprec.data.yambda_loader import DataConfig, apply_item_remap, prepare_interactions
from grouprec.experiment import resolve_paths
from grouprec.utils.run_meta import write_run_json


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    audio_dir = paths["audio"]
    audio_dir.mkdir(parents=True, exist_ok=True)

    with (paths["scorer"] / "item_id_to_idx.pkl").open("rb") as f:
        item_id_to_idx = pickle.load(f)

    embeddings = _item_embeddings(item_id_to_idx, audio_dir, args.profiles_only)
    metrics = _build_profiles(cfg, embeddings, audio_dir, item_id_to_idx)
    metrics["item_audio_coverage"] = float((np.linalg.norm(embeddings[1:], axis=1) > 0).mean())

    dump_resolved(cfg, audio_dir / "config.resolved.json")
    print(write_run_json(audio_dir, config=cfg.get("audio", {}), metrics=metrics))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profiles-only", action="store_true", help="не качать parquet с HF")
    return parser.parse_args()


def _item_embeddings(
    item_id_to_idx: dict[int, int], audio_dir: Path, profiles_only: bool
) -> np.ndarray:
    out_path = audio_dir / "embeddings.npy"
    if profiles_only or out_path.exists():
        return np.load(out_path)
    return extract_audio_subset(item_id_to_idx, out_path, use_normalized=False)


def _build_profiles(
    cfg: dict[str, Any],
    embeddings: np.ndarray,
    audio_dir: Path,
    item_id_to_idx: dict[int, int],
) -> dict[str, Any]:
    """Профили считаются по train-истории — val/test в них протекать не должны."""
    df = apply_item_remap(prepare_interactions(build(DataConfig, cfg.get("data"))), item_id_to_idx)
    train_df, _, _ = global_temporal_split(df, build(SplitConfig, cfg.get("split")))

    profiles, uid_to_row, user_audio_valid = build_user_audio_profiles(train_df, embeddings)
    np.save(audio_dir / "user_profiles.npy", profiles)
    np.save(audio_dir / "user_audio_valid.npy", user_audio_valid)
    with (audio_dir / "uid_to_row.pkl").open("wb") as f:
        pickle.dump(uid_to_row, f)

    with_audio = int(user_audio_valid.sum())
    print(f"profiles: {profiles.shape}, с аудио {with_audio:,}/{len(uid_to_row):,}")
    return {
        "n_users": len(uid_to_row),
        "user_audio_coverage": float(user_audio_valid.mean()),
    }


if __name__ == "__main__":
    main()
