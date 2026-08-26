"""Обучение групповых агрегаторов по конфигу.

uv run python research/scripts/train_aggregators.py --config research/configs/aggregators_50m.yaml
uv run python research/scripts/train_aggregators.py --config ... --models audio_agree
"""

from __future__ import annotations

import argparse
import inspect
import pickle
from dataclasses import replace
from typing import Any

import torch
from grouprec import aggregators as agg_module
from grouprec.config import build, dump_resolved, load_config
from grouprec.data.group_synthesis import GroupSynthConfig, synthesize_group_splits
from grouprec.eval.group_eval import build_group_samples, test_targets_from_df
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
    ctx = load_group_context(cfg)

    groups = _synthesize_and_save(cfg, ctx)
    samples = {
        split: _build_samples(cfg, ctx, groups[split], getattr(ctx, f"{split}_df"))
        for split in ("train", "val")
    }
    pop_counts = compute_pop_counts(
        ctx.train_df, n_items=ctx.n_items, smoothing=cfg.get("pop_smoothing", 0.75)
    )

    selected = args.models or list(cfg["models"])
    for name in selected:
        _train_one(name, cfg, ctx, samples, pop_counts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="*", help="подмножество ключей секции models")
    return parser.parse_args()


def _synthesize_and_save(cfg: dict[str, Any], ctx: GroupContext) -> dict[str, list[list[int]]]:
    """Один seed-split групп на все методы; test-группы тоже здесь — их читает eval."""
    synth_cfg = build(GroupSynthConfig, cfg.get("groups"))
    groups = synthesize_group_splits(ctx.user_pool, synth_cfg)
    out = ctx.paths["aggregators"] / "groups_split.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({**{f"{k}_groups": v for k, v in groups.items()}, "config": synth_cfg}, f)
    print("groups: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()) + f" -> {out}")
    return groups


def _build_samples(cfg: dict[str, Any], ctx: GroupContext, groups: list[list[int]], df) -> list:
    samples, stats = build_group_samples(
        groups, ctx.user_topk, test_targets_from_df(df), **cfg.get("samples", {})
    )
    print(f"  samples: kept {stats['n_kept']}/{stats['n_input_groups']}")
    return samples


def _train_one(
    name: str,
    cfg: dict[str, Any],
    ctx: GroupContext,
    samples: dict[str, list],
    pop_counts: Any,
) -> None:
    spec = cfg["models"][name]
    out_dir = ctx.paths["aggregators"] / name
    train_cfg = replace(
        build(GroupTrainConfig, cfg.get("train")), out_dir=str(out_dir), **spec.get("train", {})
    )

    torch.manual_seed(train_cfg.seed)
    model = _build_model(spec, ctx)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name} ({spec['cls']}, {n_params:,} params) ===")

    trainer = GroupAggregatorTrainer(
        aggregator=model,
        cfg=train_cfg,
        user_score_lookup=ctx.user_score_lookup,
        pop_counts=pop_counts,
        item_audio=ctx.item_audio,
        user_profiles=ctx.user_profiles,
        uid_to_row=ctx.uid_to_row,
    )
    result = trainer.fit(samples["train"], samples["val"], verbose=True)

    dump_resolved(cfg, out_dir / "config.resolved.json")
    metrics = {k: v for k, v in result.items() if k != "history"} | {"n_params": n_params}
    print(f"{name}: {write_run_json(out_dir, config=train_cfg, metrics=metrics)}")


def _build_model(spec: dict[str, Any], ctx: GroupContext) -> torch.nn.Module:
    """`uid_list`/`num_items` известны только в рантайме — подставляем тем, кто их принимает."""
    cls = getattr(agg_module, spec["cls"])
    params = dict(spec.get("params", {}))
    accepted = inspect.signature(cls).parameters
    runtime = {"uid_list": ctx.user_pool, "num_items": ctx.n_items}
    params.update({k: v for k, v in runtime.items() if k in accepted})
    return cls(**params)


if __name__ == "__main__":
    main()
