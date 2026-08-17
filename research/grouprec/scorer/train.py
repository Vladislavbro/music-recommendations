"""Training loop for gSASRec (Phase 1, per-user scorer).

Сэмплирование sequence-to-sequence:
- Для каждого пользователя берём его train-последовательность
  (отсортированную по timestamp, последние max_seq_len+1 events).
- input  = seq[:-1]   (left-padded до max_seq_len, items in 1..n_items, 0=pad)
- target = seq[1:]    (left-padded нулями; loss считается только по непаддингу)
- За одну эпоху каждая пользовательская последовательность видится один раз.

Negatives:
- Семплируются по popularity-распределению из train (стандарт word2vec/SASRec).
- n_neg штук на каждый позиционный шаг.

Loss:
- gBCE из `gbce_loss.py` с `t = 0.75` (Petrov & Macdonald 2023).
- Логиты считаются как dot product hidden state и embedding'а кандидата.
- Loss усредняется только по non-padding позициям через flat-маску.

Validation:
- NDCG@10 на тех val-юзерах, у которых есть и train-история, и val-target.
- Полный каталог (n_items), история юзера исключается из ранжирования.
- Считаем батчами на GPU; item_embeddings один раз материализуются на device.

Модель сохраняется по best val NDCG@10 в `out_dir/best.pt` + конфиг в `out_dir/config.json`.
Журнал лосса/метрик пишется в `out_dir/metrics.csv`.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .gbce_loss import gbce_loss
from .gsasrec import PAD_IDX, GSASRec

# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def build_user_sequences(
    df: pd.DataFrame,
    max_seq_len: int,
    item_col: str = "item_idx",
) -> dict[int, np.ndarray]:
    """Per-user list of last (max_seq_len+1) item indices, ordered by timestamp asc.

    `df` must contain columns `uid`, `timestamp`, and `item_col`.
    Users with fewer than 2 events are dropped (нет цели для seq2seq).
    """
    needed = {"uid", "timestamp", item_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"build_user_sequences: missing columns {missing}")

    sorted_df = df.sort_values(["uid", "timestamp"], kind="mergesort")
    keep_n = max_seq_len + 1

    out: dict[int, np.ndarray] = {}
    for uid, group in sorted_df.groupby("uid", sort=False):
        seq = group[item_col].to_numpy()
        if seq.shape[0] < 2:
            continue
        if seq.shape[0] > keep_n:
            seq = seq[-keep_n:]
        out[int(uid)] = seq.astype(np.int64, copy=False)
    return out


def compute_item_popularity(
    df: pd.DataFrame,
    n_items: int,
    item_col: str = "item_idx",
    smoothing: float = 0.75,
) -> np.ndarray:
    """Popularity^smoothing distribution over items 1..n_items (size n_items+1).

    Index 0 (padding) gets probability 0. Сглаживание `smoothing=0.75`
    стандартно для word2vec — снижает доминирование топовых items.
    """
    counts = np.zeros(n_items + 1, dtype=np.float64)
    bincount = np.bincount(df[item_col].to_numpy())
    counts[: bincount.shape[0]] = bincount
    counts[PAD_IDX] = 0.0
    if smoothing != 1.0:
        counts = counts**smoothing
    total = counts.sum()
    if total <= 0:
        raise ValueError("compute_item_popularity: empty popularity")
    return counts / total


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------


class SeqTrainDataset(Dataset):
    """Wrap dict[uid -> np.ndarray] в индексируемый датасет."""

    def __init__(self, sequences: dict[int, np.ndarray], max_seq_len: int) -> None:
        self.uids = np.array(sorted(sequences.keys()), dtype=np.int64)
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.uids.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.sequences[int(self.uids[idx])]


def _left_pad(seq: np.ndarray, length: int) -> np.ndarray:
    if seq.shape[0] >= length:
        return seq[-length:].astype(np.int64, copy=False)
    out = np.zeros(length, dtype=np.int64)
    out[length - seq.shape[0] :] = seq
    return out


def make_train_collate(max_seq_len: int):
    """Collate function: производит inputs, targets [B, L] с левым паддингом."""
    L = max_seq_len

    def collate(batch: list[np.ndarray]) -> dict[str, torch.Tensor]:
        B = len(batch)
        inputs = np.zeros((B, L), dtype=np.int64)
        targets = np.zeros((B, L), dtype=np.int64)
        for i, seq in enumerate(batch):
            # input = seq[:-1], target = seq[1:]; both left-padded to L
            inp = seq[:-1]
            tgt = seq[1:]
            if inp.shape[0] > L:
                inp = inp[-L:]
                tgt = tgt[-L:]
            inputs[i, L - inp.shape[0] :] = inp
            targets[i, L - tgt.shape[0] :] = tgt
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
        }

    return collate


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------


class PopularityNegativeSampler:
    """Сэмплер негативов по сглаженной popularity-распределению.

    Использует numpy (быстрее torch.multinomial для больших каталогов и CPU rng).
    Возможные коллизии с позитивом игнорируются: при больших каталогах
    P(коллизии) ≈ n_neg/n_items ≪ 1, эффект на gBCE пренебрежим.
    """

    def __init__(self, probs: np.ndarray, seed: int = 0) -> None:
        self.probs = probs.astype(np.float64)
        self.rng = np.random.default_rng(seed)
        self._n = probs.shape[0]

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        n = int(np.prod(shape))
        flat = self.rng.choice(self._n, size=n, replace=True, p=self.probs)
        return torch.from_numpy(flat).long().reshape(shape)


class MixedNegativeSampler:
    """Смешанный сэмплер: часть негативов из popularity, часть uniform.

    Мотивация: чистая popularity-выборка даёт каталог-инвариантные «лёгкие»
    негативы (топ поп-треки), что приводит к жанровому shortcut и нулевому
    val NDCG. Подмешивая uniform-сэмплы из всего каталога (включая хвост),
    мы даём модели fine-grained negatives — треки из того же кластера
    вкуса, что и позитив. Без этого gSASRec на больших каталогах не учит
    persona-level next-item.

    `mix_uniform` ∈ [0, 1] — доля uniform; mix=0 эквивалент `PopularityNegativeSampler`,
    mix=1 — чистый uniform.
    """

    def __init__(
        self,
        probs: np.ndarray,
        n_items: int,
        mix_uniform: float = 0.5,
        seed: int = 0,
    ) -> None:
        assert 0.0 <= mix_uniform <= 1.0
        self.probs = probs.astype(np.float64)
        self.n_items = int(n_items)
        self.mix_uniform = float(mix_uniform)
        self.rng = np.random.default_rng(seed)

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        n = int(np.prod(shape))
        n_uni = int(round(n * self.mix_uniform))
        n_pop = n - n_uni
        parts = []
        if n_pop > 0:
            parts.append(
                self.rng.choice(self.probs.shape[0], size=n_pop, replace=True, p=self.probs)
            )
        if n_uni > 0:
            # uniform по items 1..n_items (PAD_IDX=0 исключаем)
            parts.append(self.rng.integers(1, self.n_items + 1, size=n_uni))
        flat = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self.rng.shuffle(flat)
        return torch.from_numpy(flat).long().reshape(shape)


# ---------------------------------------------------------------------------
# Validation NDCG@K
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_ndcg(
    model: GSASRec,
    train_sequences: dict[int, np.ndarray],
    val_targets: dict[int, set[int]],
    max_seq_len: int,
    n_items: int,
    k: int = 10,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """Full-catalog NDCG@k. История пользователя исключается из ранжирования."""
    model.eval()
    item_emb = model.item_embeddings.weight  # [n_items+1, H]

    common_uids = [u for u in val_targets.keys() if u in train_sequences]
    if not common_uids:
        return float("nan")

    total_ndcg = 0.0
    n_users = 0

    for start in range(0, len(common_uids), batch_size):
        chunk = common_uids[start : start + batch_size]
        B = len(chunk)
        seq_batch = np.zeros((B, max_seq_len), dtype=np.int64)
        for i, u in enumerate(chunk):
            s = train_sequences[u]
            seq_batch[i] = _left_pad(s, max_seq_len)
        seq_t = torch.from_numpy(seq_batch).to(device)

        h = model(seq_t)  # [B, L, H]
        last = h[:, -1, :]  # [B, H]
        scores = last @ item_emb.T  # [B, n_items+1]
        scores[:, PAD_IDX] = -float("inf")

        topk = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()  # [B, k]

        for i, u in enumerate(chunk):
            relevant = val_targets[u]
            if not relevant:
                continue
            n_rel = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(j + 2) for j in range(n_rel))
            dcg = 0.0
            for rank, item in enumerate(topk[i]):
                if int(item) in relevant:
                    dcg += 1.0 / math.log2(rank + 2)
            total_ndcg += dcg / idcg if idcg > 0 else 0.0
            n_users += 1

    return total_ndcg / n_users if n_users > 0 else float("nan")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n_items: int
    max_seq_len: int = 200
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.2
    n_neg: int = 512
    gbce_t: float = 0.75
    mix_uniform: float = 0.5  # доля uniform-негативов; 0 = pure popularity
    batch_size: int = 128
    eval_batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 30
    early_stop_patience: int = 5
    seed: int = 42
    pop_smoothing: float = 0.75
    eval_k: int = 10
    log_every_steps: int = 50
    out_dir: str = "artifacts/gsasrec"
    device: str = "cuda"


def train_gsasrec(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainConfig,
) -> dict:
    """Обучение gSASRec на train_df, валидация по val_df. Сохраняет best checkpoint.

    Возвращает dict с финальными метриками и путём до чекпоинта.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    # ---- prepare data ----
    train_seqs = build_user_sequences(train_df, cfg.max_seq_len)
    if not train_seqs:
        raise ValueError("train_gsasrec: no users with >=2 train events")

    val_targets: dict[int, set[int]] = {
        int(uid): set(int(x) for x in g["item_idx"].to_numpy())
        for uid, g in val_df.groupby("uid", sort=False)
    }

    pop = compute_item_popularity(train_df, cfg.n_items, smoothing=cfg.pop_smoothing)
    neg_sampler = MixedNegativeSampler(
        pop, n_items=cfg.n_items, mix_uniform=cfg.mix_uniform, seed=cfg.seed
    )

    dataset = SeqTrainDataset(train_seqs, cfg.max_seq_len)
    collate = make_train_collate(cfg.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )

    # ---- model ----
    model = GSASRec(
        n_items=cfg.n_items,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metrics_path = out_dir / "metrics.csv"
    best_path = out_dir / "best.pt"
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2))

    best_ndcg = -float("inf")
    epochs_no_improve = 0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", f"val_ndcg@{cfg.eval_k}", "epoch_seconds"])

    for epoch in range(cfg.n_epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0

        for step, batch in enumerate(loader):
            inputs = batch["inputs"].to(device, non_blocking=True)  # [B, L]
            targets = batch["targets"].to(device, non_blocking=True)  # [B, L]

            mask = targets != PAD_IDX  # [B, L]
            n_pos = int(mask.sum().item())
            if n_pos == 0:
                continue

            hidden = model(inputs)  # [B, L, H]

            flat_h = hidden[mask]  # [N, H]
            flat_pos = targets[mask]  # [N]
            negs = neg_sampler.sample((flat_h.shape[0], cfg.n_neg)).to(device)  # [N, n_neg]

            pos_emb = model.item_embeddings(flat_pos)  # [N, H]
            neg_emb = model.item_embeddings(negs)  # [N, n_neg, H]

            pos_logits = (flat_h * pos_emb).sum(dim=-1)  # [N]
            neg_logits = torch.einsum("nh,nkh->nk", flat_h, neg_emb)  # [N, n_neg]

            loss = gbce_loss(
                pos_logits, neg_logits, n_items=cfg.n_items, n_neg=cfg.n_neg, t=cfg.gbce_t
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_sum += float(loss.item())
            n_batches += 1

            if cfg.log_every_steps and (step + 1) % cfg.log_every_steps == 0:
                print(
                    f"  epoch {epoch} step {step + 1}/{len(loader)} loss={loss_sum / n_batches:.4f}"
                )

        avg_loss = loss_sum / max(n_batches, 1)
        val_ndcg = evaluate_ndcg(
            model,
            train_sequences=train_seqs,
            val_targets=val_targets,
            max_seq_len=cfg.max_seq_len,
            n_items=cfg.n_items,
            k=cfg.eval_k,
            batch_size=cfg.eval_batch_size,
            device=str(device),
        )
        dt = time.time() - t0
        print(
            f"epoch {epoch}: loss={avg_loss:.4f} "
            f"val_ndcg@{cfg.eval_k}={val_ndcg:.4f} time={dt:.1f}s"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{val_ndcg:.6f}", f"{dt:.2f}"])

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_ndcg": val_ndcg,
                },
                best_path,
            )
            print(f"  -> new best, saved {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"early stop at epoch {epoch}, best ndcg@{cfg.eval_k}={best_ndcg:.4f}")
                break

    return {
        "best_ndcg": best_ndcg,
        "checkpoint": str(best_path),
        "metrics_csv": str(metrics_path),
        "config_json": str(config_path),
    }
