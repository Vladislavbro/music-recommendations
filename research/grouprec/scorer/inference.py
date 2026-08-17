"""Кэширование персональных скоров gSASRec → parquet (Phase 1, Step I).

Назначение: после обучения скорера прогнать его один раз для всех нужных
пользователей и сохранить топ-K (item_idx, score) на пользователя. Дальнейшие
эксперименты (Phase 2 — агрегаторы) читают этот кэш и не дёргают gSASRec на
каждом батче.

Формат выхода: одна большая parquet-таблица с колонками
    uid       int64
    item_idx  int64
    score     float32
    rank      int32   (0..K-1, по убыванию score)

Семантика:
- Для каждого user'а строится последовательность из его train-истории
  (последние max_seq_len events, отсортированные по timestamp asc) и берётся
  hidden state последней позиции.
- Скоринг ведётся по полному каталогу (n_items), padding-индекс 0 исключается.
- По умолчанию из ранжирования исключаются items, которые user уже слышал в
  train (`exclude_history=True`) — стандарт next-item оценки.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .gsasrec import PAD_IDX, GSASRec
from .train import _left_pad


@dataclass
class InferenceConfig:
    K: int = 200
    batch_size: int = 64
    exclude_history: bool = True
    device: str = "cuda"


def load_checkpoint(checkpoint_path: str | Path, device: str = "cuda") -> tuple[GSASRec, dict]:
    """Восстанавливает модель из чекпоинта, сохранённого `train_gsasrec`."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GSASRec(
        n_items=cfg["n_items"],
        hidden_dim=cfg["hidden_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def cache_user_scores(
    model: GSASRec,
    sequences: dict[int, np.ndarray],
    max_seq_len: int,
    n_items: int,
    out_path: str | Path,
    cfg: InferenceConfig = InferenceConfig(),  # noqa: B008 — frozen-ish конфиг, мутаций нет
) -> Path:
    """Прогоняет модель для всех users в `sequences`, сохраняет топ-K в parquet.

    Args:
        model: обученный GSASRec (`.eval()` ставится автоматически).
        sequences: dict[uid] -> np.ndarray истории item_idx (как в train.py).
        max_seq_len: длина окна, должна совпадать с тренировочной.
        n_items: размер каталога (для размерности скоринга).
        out_path: путь до .parquet.
        cfg: параметры инференса (K, batch_size, exclude_history, device).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = model.to(device)
    model.eval()

    item_emb = model.item_embeddings.weight  # [n_items+1, H]
    uids = sorted(sequences.keys())
    K = cfg.K

    chunks: list[pd.DataFrame] = []

    for start in range(0, len(uids), cfg.batch_size):
        batch_uids = uids[start : start + cfg.batch_size]
        B = len(batch_uids)
        seq_batch = np.zeros((B, max_seq_len), dtype=np.int64)
        for i, u in enumerate(batch_uids):
            seq_batch[i] = _left_pad(sequences[u], max_seq_len)
        seq_t = torch.from_numpy(seq_batch).to(device)

        h = model(seq_t)  # [B, L, H]
        last = h[:, -1, :]  # [B, H]
        scores = last @ item_emb.T  # [B, n_items+1]
        scores[:, PAD_IDX] = -float("inf")

        if cfg.exclude_history:
            for i, u in enumerate(batch_uids):
                scores[i, sequences[u]] = -float("inf")

        topk = torch.topk(scores, k=K, dim=1)
        top_scores = topk.values.cpu().numpy()  # [B, K] float32
        top_items = topk.indices.cpu().numpy()  # [B, K] int64

        uid_col = np.repeat(np.array(batch_uids, dtype=np.int64), K)
        rank_col = np.tile(np.arange(K, dtype=np.int32), B)
        chunks.append(
            pd.DataFrame(
                {
                    "uid": uid_col,
                    "item_idx": top_items.reshape(-1),
                    "score": top_scores.reshape(-1).astype(np.float32),
                    "rank": rank_col,
                }
            )
        )

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(out_path, index=False)
    return out_path
