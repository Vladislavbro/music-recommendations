"""Audio embeddings subset extraction (Phase 2, шаг 2) +
аудиопрофили пользователей (Phase 2, шаг 4).

`embeddings.parquet` (14 GB, 7.72M items × 128) лежит на HF. Качаем целиком
в кэш `huggingface_hub`, читаем две колонки в pyarrow Table, фильтруем по
`item_id_to_idx` из Phase 1 и сохраняем плотный `[n_items+1, 128]` float32
(row 0 — PAD). Запускать на Colab (95 GB RAM, быстрая сеть).

`build_user_audio_profiles` усредняет аудиоэмбеддинги по listen-истории
пользователя — это `a_bar_u` для AudioAGREE / GroupCrossAttention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    print("[audio] downloading embeddings.parquet from HF ...")
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
    idx_arr = np.array([item_id_to_idx[int(x)] for x in sel_ids], dtype=np.int64)
    out[idx_arr] = sel_emb

    missing = len(item_id_to_idx) - sel_ids.shape[0]
    if missing:
        print(f"[audio] WARNING: {missing} item_idx без эмбеддинга (нули)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    print(f"[audio] saved {out.shape} → {output_path} ({out.nbytes / 2**20:.1f} MB)")
    return out


def build_user_audio_profiles(
    train_listens: pd.DataFrame,
    item_embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    """Аудиопрофиль `a_bar_u` = mean аудиоэмбеддингов треков из train-истории
    пользователя. Усредняем только по items с непустым эмбеддингом
    (`audio_valid`).

    Args:
        train_listens: DataFrame с колонками `uid` и `item_idx` (после
            `apply_item_remap`); только train, listen+, min_pop≥5.
        item_embeddings: массив `[n_items+1, 128]` float32 (включая PAD-row 0).
            `audio_valid` определяется как `norm > 0` (row 0 → False).

    Returns:
        (profiles, uid_to_row, user_audio_valid):
          * `profiles` — `[n_users, 128]` float32, mean по valid items
            истории. Юзеры с 0 valid items → zero-row.
          * `uid_to_row` — dict uid → row index in `profiles`.
          * `user_audio_valid` — `[n_users]` bool, True если у юзера ≥1
            valid item в истории.
    """
    audio_valid = np.linalg.norm(item_embeddings, axis=1) > 0  # [n_items+1]

    df = train_listens[["uid", "item_idx"]].copy()
    df["valid"] = audio_valid[df["item_idx"].to_numpy()]
    df = df[df["valid"]].drop(columns="valid")

    uids_sorted = np.sort(train_listens["uid"].unique())
    uid_to_row = {int(u): i for i, u in enumerate(uids_sorted)}
    n_users = len(uid_to_row)
    profiles = np.zeros((n_users, EMBED_DIM), dtype=np.float32)
    user_audio_valid = np.zeros(n_users, dtype=bool)

    # group-by uid, накапливаем сумму и count, делим
    if len(df):
        rows = np.fromiter(
            (uid_to_row[int(u)] for u in df["uid"].to_numpy()),
            dtype=np.int64,
            count=len(df),
        )
        items = df["item_idx"].to_numpy()
        np.add.at(profiles, rows, item_embeddings[items])
        counts = np.bincount(rows, minlength=n_users).astype(np.int64)
        mask = counts > 0
        profiles[mask] /= counts[mask, None]
        user_audio_valid = mask

    print(
        f"[audio] user profiles: {n_users} users, "
        f"with audio: {int(user_audio_valid.sum())} "
        f"({100 * user_audio_valid.mean():.2f}%), "
        f"norm mean (valid): {np.linalg.norm(profiles[user_audio_valid], axis=1).mean():.3f}"
    )
    return profiles, uid_to_row, user_audio_valid
