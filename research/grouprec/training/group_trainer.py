"""GroupAggregatorTrainer — общий цикл обучения групповых агрегаторов (Phase 2, шаг 6).

Контекст
--------
Скорер заморожен (см. Phase 1), все 4 агрегатора (ID-AGREE, GroupIM, Audio-AGREE,
GroupCrossAttn) тренируются поверх кэша `scores.parquet`. Здесь — общая обвязка:
датасет, collate с паддингом, fit-цикл с BPR + опциональный регуляризатор,
валидация по NDCG@K через `grouprec.eval.group_eval`.

Контракт forward у агрегатора (расширение CLAUDE.md §4, документировано в
phase_2_log):

```
aggregator(
    group_user_ids:        LongTensor[B, G_max]
    candidate_ids:         LongTensor[B, C_max]
    per_user_scores:       FloatTensor[B, G_max, C_max]   # 0 для (a) item ∉ top-K
                                                          # и (b) pad-позиций
    audio_embeds_items:    FloatTensor[B, C_max, d_a] | None
    audio_profiles_users:  FloatTensor[B, G_max, d_a] | None
    group_mask:            BoolTensor[B, G_max]            # True для реальных членов
    candidate_mask:        BoolTensor[B, C_max]            # True для реальных кандидатов
) -> FloatTensor[B, C_max]
```

Решения шага 6 (см. phase_2_log):
- `per_user_scores` fill = 0.0 для item ∉ top-K юзера и для pad-позиций.
- Negatives — popularity-weighted из `C_G \\ targets`, popularity^0.75 по train,
  как в Phase 1 (`compute_item_popularity`).
- Pad-кандидаты получают score=-inf после forward → не выигрывают ranking
  и не выбираются как negatives (они исключены ещё до forward).
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..eval.group_eval import GroupSample, evaluate_aggregator_scores

# ---------------------------------------------------------------------------
# Per-user score lookup
# ---------------------------------------------------------------------------

ItemArr = np.ndarray
ScoreArr = np.ndarray
UserScoreLookup = dict[int, tuple[ItemArr, ScoreArr]]


def build_user_score_lookup(scores_df) -> UserScoreLookup:
    """Из `scores.parquet` (uid, item_idx, score, rank) собирает per-user lookup.

    Для каждого uid возвращает пару отсортированных по item_idx массивов
    `(items, scores)` — удобно для `np.searchsorted` в `lookup_per_user_scores`.
    """
    required = {"uid", "item_idx", "score"}
    if not required.issubset(scores_df.columns):
        raise ValueError(f"scores_df must have columns {required}, got {set(scores_df.columns)}")
    out: UserScoreLookup = {}
    for uid, g in scores_df.groupby("uid", sort=False):
        items = g["item_idx"].to_numpy(dtype=np.int64)
        scores = g["score"].to_numpy(dtype=np.float32)
        order = np.argsort(items, kind="stable")
        out[int(uid)] = (items[order], scores[order])
    return out


def lookup_per_user_scores(
    lookup: UserScoreLookup,
    uid: int,
    candidates: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Vectorized lookup: возвращает scores для `candidates`, item ∉ top-K → `fill`."""
    entry = lookup.get(int(uid))
    if entry is None:
        return np.full(candidates.shape[0], fill, dtype=np.float32)
    items, scores = entry
    if items.size == 0:
        return np.full(candidates.shape[0], fill, dtype=np.float32)
    idx = np.searchsorted(items, candidates)
    clip = np.clip(idx, 0, items.size - 1)
    hit = (idx < items.size) & (items[clip] == candidates)
    return np.where(hit, scores[clip], np.float32(fill)).astype(np.float32, copy=False)


def compute_pop_counts(
    train_df,
    n_items: int,
    item_col: str = "item_idx",
    smoothing: float = 0.75,
) -> np.ndarray:
    """Сглаженная popularity-фракция размера `n_items+1` (idx 0 = PAD, p=0)."""
    counts = np.zeros(n_items + 1, dtype=np.float64)
    bincount = np.bincount(train_df[item_col].to_numpy())
    counts[: bincount.shape[0]] = bincount
    counts[0] = 0.0
    if smoothing != 1.0:
        counts = counts**smoothing
    total = counts.sum()
    if total <= 0:
        raise ValueError("compute_pop_counts: empty popularity")
    return (counts / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class GroupEvalDataset(Dataset):
    """Eval-датасет: только тензоры, без позитива/негатива."""

    def __init__(
        self,
        samples: list[GroupSample],
        user_score_lookup: UserScoreLookup,
        item_audio: np.ndarray | None = None,
        user_profiles: np.ndarray | None = None,
        uid_to_row: Mapping[int, int] | None = None,
        fill_score: float = 0.0,
    ) -> None:
        self.samples = samples
        self.user_score_lookup = user_score_lookup
        self.item_audio = item_audio
        self.user_profiles = user_profiles
        self.uid_to_row = dict(uid_to_row) if uid_to_row is not None else None
        self.fill_score = float(fill_score)
        self.d_audio = int(item_audio.shape[1]) if item_audio is not None else 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        cands = np.asarray(s.candidates, dtype=np.int64)
        members = np.asarray(s.members, dtype=np.int64)
        G = members.shape[0]
        C = cands.shape[0]

        per_user = np.zeros((G, C), dtype=np.float32)
        for gi, uid in enumerate(members):
            per_user[gi] = lookup_per_user_scores(
                self.user_score_lookup, int(uid), cands, fill=self.fill_score
            )

        if self.item_audio is not None:
            item_aud = self.item_audio[cands]  # [C, d_a]
        else:
            item_aud = np.zeros((C, self.d_audio), dtype=np.float32)

        if self.user_profiles is not None and self.uid_to_row is not None:
            user_aud = np.zeros((G, self.d_audio), dtype=np.float32)
            for gi, uid in enumerate(members):
                row = self.uid_to_row.get(int(uid))
                if row is not None:
                    user_aud[gi] = self.user_profiles[row]
        else:
            user_aud = np.zeros((G, self.d_audio), dtype=np.float32)

        return {
            "members": members,
            "candidates": cands,
            "per_user_scores": per_user,
            "item_audio": item_aud,
            "user_audio": user_aud,
            "G": G,
            "C": C,
            "sample_idx": idx,
        }


class GroupTrainDataset(GroupEvalDataset):
    """Train-датасет: каждый __getitem__ дополнительно семплирует pos + n_neg негативов.

    Pos — равномерно из `sample.targets`. Neg — popularity-weighted (counts^0.75
    по train, см. `compute_pop_counts`) из `candidates \\ targets`.
    """

    def __init__(
        self,
        samples: list[GroupSample],
        user_score_lookup: UserScoreLookup,
        pop_counts: np.ndarray,
        n_neg_per_pos: int = 4,
        item_audio: np.ndarray | None = None,
        user_profiles: np.ndarray | None = None,
        uid_to_row: Mapping[int, int] | None = None,
        fill_score: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            samples=samples,
            user_score_lookup=user_score_lookup,
            item_audio=item_audio,
            user_profiles=user_profiles,
            uid_to_row=uid_to_row,
            fill_score=fill_score,
        )
        self.pop_counts = pop_counts.astype(np.float64, copy=False)
        self.n_neg = int(n_neg_per_pos)
        self.rng = np.random.default_rng(seed)

        # Precompute: для каждого sample — positions of targets within candidates.
        # Также — pop weights для (candidates \ targets) уже нормированные.
        self._neg_weights: list[np.ndarray | None] = []
        self._pos_positions: list[np.ndarray] = []
        for s in samples:
            cands = s.candidates
            targets = s.targets
            pos_in_cands = np.searchsorted(cands, targets)  # cands sorted (от np.unique)
            self._pos_positions.append(pos_in_cands.astype(np.int64, copy=False))

            w = self.pop_counts[cands].copy()
            w[pos_in_cands] = 0.0  # исключаем targets
            tot = w.sum()
            if tot <= 0:
                self._neg_weights.append(None)  # фолбэк на uniform по non-target
            else:
                self._neg_weights.append((w / tot).astype(np.float64, copy=False))

    def __getitem__(self, idx: int) -> dict:
        base = super().__getitem__(idx)
        C = base["C"]

        pos_positions = self._pos_positions[idx]
        if pos_positions.size == 0:
            # Не должно случаться при drop_empty=True, но на всякий — берём 0 и помечаем.
            pos_idx = 0
            valid = False
        else:
            pos_idx = int(self.rng.choice(pos_positions))
            valid = True

        w = self._neg_weights[idx]
        n_pool = C - pos_positions.size
        if w is None or n_pool <= 0:
            # фолбэк: uniform по всем позициям кроме targets
            mask = np.ones(C, dtype=bool)
            mask[pos_positions] = False
            available = np.flatnonzero(mask)
            if available.size == 0:
                neg_idx = np.zeros(self.n_neg, dtype=np.int64)
                valid = False
            else:
                neg_idx = self.rng.choice(available, size=self.n_neg, replace=True).astype(np.int64)
        else:
            neg_idx = self.rng.choice(C, size=self.n_neg, replace=True, p=w).astype(np.int64)

        base["pos_idx"] = np.int64(pos_idx)
        base["neg_idx"] = neg_idx
        base["valid"] = bool(valid)
        return base


# ---------------------------------------------------------------------------
# Collate (с паддингом до batch-max по G и C)
# ---------------------------------------------------------------------------


def collate_groups(batch: list[dict]) -> dict[str, torch.Tensor]:
    B = len(batch)
    G_max = max(b["G"] for b in batch)
    C_max = max(b["C"] for b in batch)
    d_a = batch[0]["item_audio"].shape[1]

    members = np.zeros((B, G_max), dtype=np.int64)
    candidates = np.zeros((B, C_max), dtype=np.int64)
    per_user = np.zeros((B, G_max, C_max), dtype=np.float32)
    item_aud = np.zeros((B, C_max, d_a), dtype=np.float32)
    user_aud = np.zeros((B, G_max, d_a), dtype=np.float32)
    group_mask = np.zeros((B, G_max), dtype=bool)
    cand_mask = np.zeros((B, C_max), dtype=bool)
    sample_idx = np.zeros(B, dtype=np.int64)

    is_train = "pos_idx" in batch[0]
    if is_train:
        n_neg = batch[0]["neg_idx"].shape[0]
        pos_idx = np.zeros(B, dtype=np.int64)
        neg_idx = np.zeros((B, n_neg), dtype=np.int64)
        valid = np.zeros(B, dtype=bool)

    for i, b in enumerate(batch):
        G = b["G"]
        C = b["C"]
        members[i, :G] = b["members"]
        candidates[i, :C] = b["candidates"]
        per_user[i, :G, :C] = b["per_user_scores"]
        item_aud[i, :C] = b["item_audio"]
        user_aud[i, :G] = b["user_audio"]
        group_mask[i, :G] = True
        cand_mask[i, :C] = True
        sample_idx[i] = b["sample_idx"]
        if is_train:
            pos_idx[i] = b["pos_idx"]
            neg_idx[i] = b["neg_idx"]
            valid[i] = b["valid"]

    out = {
        "members": torch.from_numpy(members),
        "candidates": torch.from_numpy(candidates),
        "per_user_scores": torch.from_numpy(per_user),
        "item_audio": torch.from_numpy(item_aud),
        "user_audio": torch.from_numpy(user_aud),
        "group_mask": torch.from_numpy(group_mask),
        "candidate_mask": torch.from_numpy(cand_mask),
        "sample_idx": torch.from_numpy(sample_idx),
    }
    if is_train:
        out["pos_idx"] = torch.from_numpy(pos_idx)
        out["neg_idx"] = torch.from_numpy(neg_idx)
        out["valid"] = torch.from_numpy(valid)
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class GroupTrainConfig:
    n_epochs: int = 20
    batch_size: int = 64
    eval_batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_neg_per_pos: int = 4
    reg_loss_weight: float = 0.0  # λ для aggregator.regularization_loss(), если есть
    eval_k: Sequence[int] = (10, 20)  # Sequence, а не tuple: список из YAML подходит как есть
    early_stop_patience: int = 5
    seed: int = 42
    log_every_steps: int = 50
    out_dir: str = "artifacts/aggregators/run"
    device: str = "cuda"
    fill_score: float = 0.0


class GroupAggregatorTrainer:
    """Обучает любой `nn.Module`, реализующий контракт `forward(...)` Phase 2.

    Скорер из Phase 1 заморожен и не передаётся сюда — его выход уже лежит в
    `user_score_lookup` (per-user топ-K). Обновляются только параметры
    переданного `aggregator`.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        cfg: GroupTrainConfig,
        user_score_lookup: UserScoreLookup,
        pop_counts: np.ndarray,
        item_audio: np.ndarray | None = None,
        user_profiles: np.ndarray | None = None,
        uid_to_row: Mapping[int, int] | None = None,
    ) -> None:
        self.aggregator = aggregator
        self.cfg = cfg
        self.user_score_lookup = user_score_lookup
        self.pop_counts = pop_counts
        self.item_audio = item_audio
        self.user_profiles = user_profiles
        self.uid_to_row = uid_to_row

        device_str = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device_str)
        self.aggregator.to(self.device)
        self.optim = torch.optim.Adam(
            [p for p in aggregator.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # -- batch step ---------------------------------------------------------

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.aggregator(
            group_user_ids=batch["members"],
            candidate_ids=batch["candidates"],
            per_user_scores=batch["per_user_scores"],
            audio_embeds_items=batch["item_audio"],
            audio_profiles_users=batch["user_audio"],
            group_mask=batch["group_mask"],
            candidate_mask=batch["candidate_mask"],
        )

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        from .bpr_loss import pairwise_bpr_loss  # локальный импорт чтобы избежать циклов

        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        scores = self._forward(batch)  # [B, C_max]

        # Pad-candidates → -inf, чтобы они не выигрывали ranking.
        scores = scores.masked_fill(~batch["candidate_mask"], float("-inf"))

        valid = batch["valid"]
        if not valid.any():
            return 0.0

        # Только валидные сэмплы (с непустыми targets) участвуют в loss.
        scores_v = scores[valid]
        pos_idx = batch["pos_idx"][valid].unsqueeze(1)  # [Bv, 1]
        neg_idx = batch["neg_idx"][valid]  # [Bv, K]

        pos_scores = scores_v.gather(1, pos_idx).squeeze(1)  # [Bv]
        neg_scores = scores_v.gather(1, neg_idx)  # [Bv, K]

        loss = pairwise_bpr_loss(pos_scores, neg_scores)

        if self.cfg.reg_loss_weight > 0.0 and hasattr(self.aggregator, "regularization_loss"):
            reg = self.aggregator.regularization_loss(batch, scores)  # type: ignore[attr-defined]
            loss = loss + self.cfg.reg_loss_weight * reg

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())

    # -- eval ---------------------------------------------------------------

    @torch.no_grad()
    def predict_group_scores(
        self,
        samples: list[GroupSample],
        batch_size: int | None = None,
    ) -> list[np.ndarray]:
        """Возвращает list[np.ndarray] длины `samples`, элементы — `[|C_G|]`."""
        if not samples:
            return []
        self.aggregator.eval()
        ds = GroupEvalDataset(
            samples=samples,
            user_score_lookup=self.user_score_lookup,
            item_audio=self.item_audio,
            user_profiles=self.user_profiles,
            uid_to_row=self.uid_to_row,
            fill_score=self.cfg.fill_score,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size or self.cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collate_groups,
            num_workers=0,
        )
        results: list[np.ndarray | None] = [None] * len(samples)
        for batch in loader:
            batch_dev = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            scores = self._forward(batch_dev)  # [B, C_max]
            scores = scores.masked_fill(~batch_dev["candidate_mask"], float("-inf"))
            sc_cpu = scores.detach().cpu().numpy()
            cmask_cpu = batch["candidate_mask"].numpy()
            sample_idx = batch["sample_idx"].numpy()
            for row in range(sc_cpu.shape[0]):
                C = int(cmask_cpu[row].sum())
                results[int(sample_idx[row])] = sc_cpu[row, :C].astype(np.float32, copy=False)
        # type: ignore[return-value]
        return [r if r is not None else np.zeros(0, dtype=np.float32) for r in results]

    def evaluate(self, samples: list[GroupSample]) -> dict:
        scores = self.predict_group_scores(samples)
        return evaluate_aggregator_scores(samples, scores, k_list=list(self.cfg.eval_k))

    # -- fit ----------------------------------------------------------------

    def fit(
        self,
        train_samples: list[GroupSample],
        val_samples: list[GroupSample],
        verbose: bool = True,
    ) -> dict:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path = out_dir / "config.json"
        config_path.write_text(json.dumps(asdict(cfg), indent=2, default=list))
        metrics_path = out_dir / "metrics.csv"
        best_path = out_dir / "best.pt"

        train_ds = GroupTrainDataset(
            samples=train_samples,
            user_score_lookup=self.user_score_lookup,
            pop_counts=self.pop_counts,
            n_neg_per_pos=cfg.n_neg_per_pos,
            item_audio=self.item_audio,
            user_profiles=self.user_profiles,
            uid_to_row=self.uid_to_row,
            fill_score=cfg.fill_score,
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_groups,
            num_workers=0,
            drop_last=False,
        )

        primary_k = cfg.eval_k[0]
        header = ["epoch", "train_loss", f"val_NDCG@{primary_k}", "epoch_seconds"]
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

        best_score = -float("inf")
        no_improve = 0
        history: list[dict] = []

        for epoch in range(cfg.n_epochs):
            self.aggregator.train()
            t0 = time.time()
            loss_sum = 0.0
            n_batches = 0
            for step, batch in enumerate(train_loader):
                loss_val = self._train_step(batch)
                loss_sum += loss_val
                n_batches += 1
                if verbose and cfg.log_every_steps and (step + 1) % cfg.log_every_steps == 0:
                    print(
                        f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                        f"loss={loss_sum / max(n_batches, 1):.4f}"
                    )

            avg_loss = loss_sum / max(n_batches, 1)
            val_metrics = self.evaluate(val_samples)
            val_primary = val_metrics[f"NDCG@{primary_k}"]
            dt = time.time() - t0

            if verbose:
                extra = " ".join(f"val_NDCG@{k}={val_metrics[f'NDCG@{k}']:.4f}" for k in cfg.eval_k)
                print(f"epoch {epoch}: loss={avg_loss:.4f} {extra} time={dt:.1f}s")
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, f"{avg_loss:.6f}", f"{val_primary:.6f}", f"{dt:.2f}"]
                )
            history.append(
                {"epoch": epoch, "train_loss": avg_loss, "val": val_metrics, "seconds": dt}
            )

            if val_primary > best_score:
                best_score = val_primary
                no_improve = 0
                torch.save(
                    {
                        "aggregator_state": self.aggregator.state_dict(),
                        "config": asdict(cfg),
                        "epoch": epoch,
                        f"val_NDCG@{primary_k}": val_primary,
                    },
                    best_path,
                )
                if verbose:
                    print(f"  -> new best, saved {best_path}")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    if verbose:
                        print(
                            f"early stop at epoch {epoch}, "
                            f"best val_NDCG@{primary_k}={best_score:.4f}"
                        )
                    break

        return {
            f"best_val_NDCG@{primary_k}": best_score,
            "checkpoint": str(best_path),
            "metrics_csv": str(metrics_path),
            "config_json": str(config_path),
            "history": history,
        }
