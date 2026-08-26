"""Оценка агрегаторов на test-группах: NDCG + bootstrap CI + paired-разности.

uv run python research/scripts/eval_groups.py --config research/configs/aggregators_50m.yaml
"""

from __future__ import annotations

import argparse
import inspect
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from grouprec import aggregators as agg_module
from grouprec.config import build, dump_resolved, load_config
from grouprec.eval.bootstrap import (
    bootstrap_indices,
    dump_per_sample,
    latex_table,
    paired_deltas,
    summarize,
    summarize_by_size,
)
from grouprec.eval.group_eval import (
    build_group_samples,
    evaluate_aggregator_scores,
    test_targets_from_df,
)
from grouprec.eval.trivial import TRIVIAL_AGGREGATORS, trivial_group_scores
from grouprec.experiment import GroupContext, load_group_context
from grouprec.training.group_trainer import (
    GroupAggregatorTrainer,
    GroupTrainConfig,
    compute_pop_counts,
)
from grouprec.utils.run_meta import write_run_json


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", {})
    ctx = load_group_context(cfg)

    samples = _load_test_samples(cfg, ctx)
    ks = list(cfg["train"]["eval_k"])
    per_sample = _score_all_methods(cfg, ctx, samples, ks)

    out_dir = ctx.paths["eval_results"]
    summary = _write_tables(per_sample, samples, ks, eval_cfg, out_dir)
    dump_resolved(cfg, out_dir / "config.resolved.json")
    metrics = summary.set_index("method").to_dict(orient="index")
    print(f"\n{write_run_json(out_dir, config=eval_cfg, metrics=metrics)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_test_samples(cfg: dict[str, Any], ctx: GroupContext) -> list:
    """Test-группы берём из зафиксированного split'а обучения, не пересинтезируем."""
    with (ctx.paths["aggregators"] / "groups_split.pkl").open("rb") as f:
        groups = pickle.load(f)["test_groups"]
    samples, stats = build_group_samples(
        groups, ctx.user_topk, test_targets_from_df(ctx.test_df), **cfg.get("samples", {})
    )
    print(f"test samples: kept {stats['n_kept']}/{stats['n_input_groups']}")
    return samples


def _score_all_methods(
    cfg: dict[str, Any], ctx: GroupContext, samples: list, ks: list[int]
) -> dict[str, dict[int, np.ndarray]]:
    per_sample = {}
    for key, spec in cfg["models"].items():
        scores = _predict_trained(key, cfg, ctx, samples)
        per_sample[spec.get("name", key)] = _report(spec.get("name", key), samples, scores, ks)
    for name, agg_fn in TRIVIAL_AGGREGATORS.items():
        fill = cfg["train"].get("fill_score", 0.0)
        scores = trivial_group_scores(samples, ctx.user_score_lookup, agg_fn, fill=fill)
        per_sample[name] = _report(name, samples, scores, ks)
    return per_sample


def _report(name: str, samples: list, scores: list, ks: list[int]) -> dict[int, np.ndarray]:
    metrics = evaluate_aggregator_scores(samples, scores, k_list=ks)
    shown = "  ".join(f"NDCG@{k}={metrics[f'NDCG@{k}']:.4f}" for k in ks)
    print(f"  {name:18s} | {shown}  n={metrics['n_samples']}")
    return {k: metrics["per_sample"][f"NDCG@{k}"] for k in ks}


def _predict_trained(
    name: str, cfg: dict[str, Any], ctx: GroupContext, samples: list
) -> list[np.ndarray]:
    spec = cfg["models"][name]
    ckpt_dir = ctx.paths["aggregators"] / name
    train_cfg = replace(build(GroupTrainConfig, cfg.get("train")), out_dir=str(ckpt_dir))

    model = _build_model(spec, ctx)
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["aggregator_state"])
    trainer = GroupAggregatorTrainer(
        aggregator=model,
        cfg=train_cfg,
        user_score_lookup=ctx.user_score_lookup,
        pop_counts=compute_pop_counts(ctx.train_df, n_items=ctx.n_items),
        item_audio=ctx.item_audio,
        user_profiles=ctx.user_profiles,
        uid_to_row=ctx.uid_to_row,
    )
    return trainer.predict_group_scores(samples)


def _build_model(spec: dict[str, Any], ctx: GroupContext) -> torch.nn.Module:
    cls = getattr(agg_module, spec["cls"])
    params = dict(spec.get("params", {}))
    accepted = inspect.signature(cls).parameters
    runtime = {"uid_list": ctx.user_pool, "num_items": ctx.n_items}
    params.update({k: v for k, v in runtime.items() if k in accepted})
    return cls(**params)


def _write_tables(
    per_sample: dict[str, dict[int, np.ndarray]],
    samples: list,
    ks: list[int],
    eval_cfg: dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = eval_cfg.get("boot_seed", 42)
    n_boot = eval_cfg.get("n_boot", 1000)
    idx = bootstrap_indices(len(samples), n_boot=n_boot, seed=seed)
    sizes = np.array([s.size for s in samples], dtype=np.int64)

    summary = summarize(per_sample, idx, ks)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summarize_by_size(per_sample, sizes, ks, n_boot=n_boot, seed=seed + 1).to_csv(
        out_dir / "summary_by_size.csv", index=False
    )
    pairs = [tuple(p) for p in eval_cfg.get("pairs", [])]
    if pairs:
        paired_deltas(per_sample, pairs, idx, ks).to_csv(out_dir / "paired.csv", index=False)
    (out_dir / "summary_table.tex").write_text(latex_table(summary, ks))
    dump_per_sample(per_sample, sizes, idx, ks, out_dir / "per_sample.npz")
    print(f"\n{summary.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    return summary


if __name__ == "__main__":
    main()
