"""Обучение per-user скорера (SASRec) и кэш топ-K скоров.

uv run python research/scripts/train_scorer.py --config research/configs/scorer_50m.yaml
uv run python research/scripts/train_scorer.py --config research/configs/smoke_local.yaml
uv run python research/scripts/train_scorer.py --config ... --cache-only  # без переобучения
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import pandas as pd
from grouprec.config import build, dump_resolved, load_config
from grouprec.data.splits import SplitConfig, global_temporal_split
from grouprec.data.yambda_loader import (
    DataConfig,
    apply_item_remap,
    build_item_id_to_idx,
    prepare_interactions,
)
from grouprec.experiment import resolve_paths
from grouprec.scorer.inference import InferenceConfig, cache_user_scores, load_checkpoint
from grouprec.scorer.train import TrainConfig, build_user_sequences, evaluate_ndcg, train_gsasrec
from grouprec.utils.caching import save_pickle
from grouprec.utils.run_meta import write_run_json
from grouprec.utils.seed import set_seed


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    paths = resolve_paths(cfg)

    splits, item_id_to_idx = _prepare_splits(cfg)
    train_cfg = replace(
        build(TrainConfig, cfg.get("train"), n_items=len(item_id_to_idx)),
        out_dir=str(paths["scorer"]),  # единственный источник пути — секция paths
    )

    if args.cache_only:
        model, _ = load_checkpoint(paths["scorer"] / "best.pt", device=train_cfg.device)
        metrics: dict[str, Any] = {}
    else:
        model, metrics = _train(splits, train_cfg, item_id_to_idx, paths)

    metrics["test_NDCG@10"] = _test_ndcg(model, splits, train_cfg)
    metrics["scores_cache"] = str(_cache_scores(cfg, model, splits, train_cfg, paths))

    dump_resolved(cfg, paths["scorer"] / "config.resolved.json")
    print(write_run_json(paths["scorer"], config=train_cfg, metrics=metrics))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-only", action="store_true", help="только пересчёт кэша скоров")
    return parser.parse_args()


def _prepare_splits(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[int, int]]:
    """Ремап строится здесь, а не читается: Phase 1 — источник `item_id_to_idx`."""
    df = prepare_interactions(build(DataConfig, cfg.get("data")))
    item_id_to_idx = build_item_id_to_idx(df)
    df = apply_item_remap(df, item_id_to_idx)
    train, val, test = global_temporal_split(df, build(SplitConfig, cfg.get("split")))
    print(f"events: train={len(train):,} val={len(val):,} test={len(test):,}")
    print(f"items: {len(item_id_to_idx):,}  users: {df['uid'].nunique():,}")
    return {"train": train, "val": val, "test": test}, item_id_to_idx


def _train(
    splits: dict[str, pd.DataFrame],
    train_cfg: TrainConfig,
    item_id_to_idx: dict[int, int],
    paths: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    result = train_gsasrec(splits["train"], splits["val"], train_cfg)
    save_pickle(item_id_to_idx, paths["scorer"] / "item_id_to_idx.pkl")
    model, _ = load_checkpoint(result["checkpoint"], device=train_cfg.device)
    return model, {k: v for k, v in result.items() if k != "history"}


def _test_ndcg(model: Any, splits: dict[str, pd.DataFrame], train_cfg: TrainConfig) -> float:
    """Диагностика скорера, не финальная метрика: она групповая (см. eval_groups.py)."""
    targets = {
        int(uid): {int(x) for x in g["item_idx"].to_numpy()}
        for uid, g in splits["test"].groupby("uid", sort=False)
    }
    ndcg = evaluate_ndcg(
        model,
        train_sequences=build_user_sequences(splits["train"], max_seq_len=train_cfg.max_seq_len),
        val_targets=targets,
        max_seq_len=train_cfg.max_seq_len,
        n_items=train_cfg.n_items,
        k=train_cfg.eval_k,
        batch_size=train_cfg.eval_batch_size,
        device=train_cfg.device,
    )
    print(f"test NDCG@{train_cfg.eval_k} = {ndcg:.4f}")
    return float(ndcg)


def _cache_scores(
    cfg: dict[str, Any],
    model: Any,
    splits: dict[str, pd.DataFrame],
    train_cfg: TrainConfig,
    paths: dict[str, Any],
) -> Any:
    inf_cfg = build(InferenceConfig, cfg.get("inference"))
    sequences = build_user_sequences(splits["train"], max_seq_len=train_cfg.max_seq_len)
    out = cache_user_scores(
        model, sequences, train_cfg.max_seq_len, train_cfg.n_items, paths["scores_cache"], inf_cfg
    )
    print(f"scores cache: {out}")
    return out


if __name__ == "__main__":
    main()
