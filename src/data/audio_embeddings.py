"""Audio embeddings subset extraction (Phase 2, шаг 2).

`embeddings.parquet` (14 GB, 7.72M items × 128) лежит на HF. Качаем целиком
в кэш `huggingface_hub`, читаем две колонки в pyarrow Table, фильтруем по
`item_id_to_idx` из Phase 1 и сохраняем плотный `[n_items+1, 128]` float32
(row 0 — PAD). Запускать на Colab (95 GB RAM, быстрая сеть).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


EMBED_DIM = 128


def extract_audio_subset(
    item_id_to_idx: dict[int, int],
    output_path: str | Path,
    use_normalized: bool = False,
) -> np.ndarray:
    """Скачивает `embeddings.parquet` с HF, фильтрует по `item_id_to_idx`,
    сохраняет плотный массив `[n_items+1, 128]` float32 в `output_path`.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    col = "normalized_embed" if use_normalized else "embed"
    print(f"[audio] downloading embeddings.parquet from HF ...")
    path = hf_hub_download(
        repo_id="yandex/yambda",
        filename="embeddings.parquet",
        repo_type="dataset",
    )
    print(f"[audio] reading columns ['item_id', {col!r}] ...")
    tbl = pq.read_table(path, columns=["item_id", col])
    ids = tbl.column("item_id").to_numpy()
    embeds = (
        tbl.column(col)
        .combine_chunks()
        .flatten()
        .to_numpy()
        .reshape(-1, EMBED_DIM)
        .astype(np.float32, copy=False)
    )
    print(f"[audio] full table: {ids.shape[0]:,} items, embeds {embeds.shape}")

    target_ids = np.fromiter(item_id_to_idx.keys(), dtype=np.uint32)
    mask = np.isin(ids, target_ids)
    sel_ids = ids[mask]
    sel_emb = embeds[mask]
    print(f"[audio] matched {sel_ids.shape[0]:,} / {len(item_id_to_idx):,}")

    max_idx = max(item_id_to_idx.values())
    out = np.zeros((max_idx + 1, EMBED_DIM), dtype=np.float32)
    idx_arr = np.array(
        [item_id_to_idx[int(x)] for x in sel_ids], dtype=np.int64
    )
    out[idx_arr] = sel_emb

    missing = len(item_id_to_idx) - sel_ids.shape[0]
    if missing:
        print(f"[audio] WARNING: {missing} item_idx без эмбеддинга (нули)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    print(f"[audio] saved {out.shape} → {output_path} ({out.nbytes / 2**20:.1f} MB)")
    return out
