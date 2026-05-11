"""Audio embeddings subset extraction (Phase 2, шаг 2).

`embeddings.parquet` (14 GB, 7.72M items) лежит на HF и локально не хранится.
Стримим row groups через `HfFileSystem` + `pyarrow`, фильтруем строки по
нашему `item_id_to_idx` (276,305 items после Phase 1 фильтра), сохраняем
плотный `[n_items+1, 128]` float32 массив, индексируемый `item_idx`
(row 0 — PAD, остаётся нулевым).

Источник на HF — datasets/yandex/yambda/embeddings.parquet
Схема: item_id uint32, embed large_list<double>, normalized_embed large_list<double>.

Запускать предпочтительно из Colab: ~14 GB сетевого трафика,
на диск пишется только результат (~135 MB).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


HF_EMBEDDINGS_PATH = "datasets/yandex/yambda/embeddings.parquet"
EMBED_DIM = 128


def extract_audio_subset(
    item_id_to_idx: dict[int, int],
    output_path: str | Path,
    use_normalized: bool = False,
    hf_token: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Стримит `embeddings.parquet` с HF, фильтрует по `item_id_to_idx`,
    сохраняет плотный массив [n_items+1, 128] float32.

    Args:
        item_id_to_idx: карта из Phase 1 (raw item_id → contiguous idx 1..N).
        output_path: куда положить .npy (например `artifacts/audio/embeddings.npy`).
        use_normalized: брать ли `normalized_embed` (L2-нормированный) вместо `embed`.
        hf_token: опциональный HF-токен, для публичного датасета не нужен.

    Returns:
        Массив shape `[n_items+1, EMBED_DIM]`, dtype float32.
        Row 0 — нули (PAD); строки для item_idx без эмбеддинга — тоже нули
        (в норме их быть не должно, см. лог `missing`).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    target_ids_np = np.fromiter(
        (int(k) for k in item_id_to_idx.keys()), dtype=np.uint32
    )
    target_ids_np.sort()
    id_to_idx_arr = np.zeros(target_ids_np.max() + 1, dtype=np.int64)
    for raw_id, idx in item_id_to_idx.items():
        id_to_idx_arr[int(raw_id)] = int(idx)
    max_idx = max(item_id_to_idx.values())

    out = np.zeros((max_idx + 1, EMBED_DIM), dtype=np.float32)
    seen = np.zeros(max_idx + 1, dtype=bool)
    seen[0] = True  # PAD — намеренно нули

    col = "normalized_embed" if use_normalized else "embed"
    fs = HfFileSystem(token=hf_token)

    with fs.open(HF_EMBEDDINGS_PATH, "rb") as f:
        pf = pq.ParquetFile(f)
        n_rg = pf.num_row_groups
        n_target = len(item_id_to_idx)
        if verbose:
            print(
                f"[audio] HF parquet: {n_rg} row groups, "
                f"{pf.metadata.num_rows:,} rows; target {n_target:,} items"
            )

        n_kept = 0
        for i in range(n_rg):
            tbl = pf.read_row_group(i, columns=["item_id", col])
            ids = tbl.column("item_id").to_numpy(zero_copy_only=False)
            mask = np.isin(ids, target_ids_np, assume_unique=False)
            n_match = int(mask.sum())
            if n_match == 0:
                if verbose:
                    print(f"[audio] rg {i + 1}/{n_rg}: 0 matches")
                continue

            sel_indices = np.where(mask)[0]
            sel_ids = ids[sel_indices]
            sel_embed_arr = (
                tbl.column(col).take(sel_indices).combine_chunks()
            )
            flat = sel_embed_arr.flatten().to_numpy(zero_copy_only=False)
            embeds = flat.reshape(n_match, EMBED_DIM).astype(np.float32, copy=False)

            target_idx = id_to_idx_arr[sel_ids]
            out[target_idx] = embeds
            seen[target_idx] = True

            n_kept += n_match
            if verbose:
                print(
                    f"[audio] rg {i + 1}/{n_rg}: +{n_match} matches, "
                    f"total {n_kept:,}/{n_target:,}"
                )

    missing = int((~seen).sum())
    if verbose:
        if missing:
            print(
                f"[audio] WARNING: {missing} item_idx без эмбеддинга "
                f"(оставлены нулевыми)"
            )
        else:
            print("[audio] все целевые item_idx покрыты")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    if verbose:
        print(
            f"[audio] saved {out.shape} float32 → {output_path} "
            f"({out.nbytes / 2**20:.1f} MB)"
        )
    return out
