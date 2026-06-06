"""Экспорт артефакта `{uid: test_item_idx[]}` для live-метрик демки (Phase E).

Ground-truth для онлайн-NDCG — это held-out test-listens YAMBDA каждого юзера.
Полный офлайн-путь (notebooks/07): load_yambda → filter_listens → filter_min_popularity
→ apply_item_remap → global_temporal_split → test_targets_from_df. Здесь срезаем
угол: нужны ТОЛЬКО test-окно (timestamp ≥ TEST_TIMESTAMP) и только demo-uid
(те, что есть в scores.parquet). Это эквивалентно полному пути, потому что:

* test-split = `timestamp ≥ TEST_TIMESTAMP`, ограниченный train-юзерами; все demo-uid
  по построению имеют train-историю (скорер на них обучен/закэширован) ⊆ train_uids;
* `filter_min_popularity` + `apply_item_remap` = «оставить item, если он в
  item_id_to_idx». Маппим через него и выкидываем непокрытые — ровно тот же набор;
* `drop_non_train_items=False` (дефолт SplitConfig) → ограничения на train-items нет.

Результат: `artifacts/test_targets/test_targets.pkl` = {int uid: np.ndarray[int64]}.
Юзеры без test-listens (после маппинга) в файл не попадают — на стороне backend
их `targets` будет пустым и группа исключается из усреднения NDCG.

Запуск (локально, YAMBDA-50m уже в HF-кэше):
    python Club-Demo/backend/export_test_targets.py
"""
from __future__ import annotations

import glob
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

# мини-зеркало констант из src.data.{splits,yambda_loader} — без тяжёлых импортов
TEST_TIMESTAMP = 25_913_600          # LAST_TIMESTAMP - 1 day
TRACK_LISTEN_THRESHOLD = 50          # played_ratio_pct ≥ 50% = реальный listen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
SCORES_PARQUET = ARTIFACTS / "user_scores_cache" / "scores.parquet"
ITEM_ID_TO_IDX_PKL = ARTIFACTS / "gsasrec" / "item_id_to_idx.pkl"
OUT_DIR = ARTIFACTS / "test_targets"
OUT_PKL = OUT_DIR / "test_targets.pkl"


def _find_yambda_parquet() -> str:
    pattern = str(
        Path.home()
        / ".cache/huggingface/hub/datasets--yandex--yambda/snapshots/*/flat/50m/multi_event.parquet"
    )
    hits = [p for p in glob.glob(pattern)]
    if not hits:
        raise FileNotFoundError(
            "YAMBDA-50m parquet не найден в HF-кэше. Сначала прогоните пайплайн "
            "(load_yambda('50m')) или скачайте датасет."
        )
    # любой из снапшотов-симлинков указывает на тот же blob
    return hits[0]


def main() -> None:
    t0 = time.time()

    demo_uids = set(
        int(u) for u in pd.read_parquet(SCORES_PARQUET, columns=["uid"])["uid"].unique()
    )
    print(f"[export] demo uids (в кэше scores): {len(demo_uids):,}")

    with open(ITEM_ID_TO_IDX_PKL, "rb") as f:
        item_id_to_idx = {int(k): int(v) for k, v in pickle.load(f).items()}
    print(f"[export] item_id_to_idx: {len(item_id_to_idx):,} items")

    pq_path = _find_yambda_parquet()
    print(f"[export] reading test window from {pq_path}")

    dataset = ds.dataset(pq_path, format="parquet")
    # фильтр на уровне сканера: test-окно + listen — это вся отсечка по объёму
    flt = (
        (pc.field("timestamp") >= TEST_TIMESTAMP)
        & (pc.field("event_type") == "listen")
        & (pc.field("played_ratio_pct") >= TRACK_LISTEN_THRESHOLD)
    )
    table = dataset.to_table(
        columns=["uid", "item_id"],
        filter=flt,
    )
    df = table.to_pandas()
    print(f"[export] test-listen rows: {len(df):,}  uids: {df['uid'].nunique():,}")

    # только demo-юзеры
    df = df[df["uid"].isin(demo_uids)]
    print(f"[export] after demo-uid filter: {len(df):,}  uids: {df['uid'].nunique():,}")

    # item_id -> item_idx; непокрытые (выкинуты min_popularity / cold) отбрасываем
    df["item_idx"] = df["item_id"].map(item_id_to_idx)
    n_before = len(df)
    df = df.dropna(subset=["item_idx"])
    df["item_idx"] = df["item_idx"].astype("int64")
    print(f"[export] mapped item_idx: kept {len(df):,}/{n_before:,} rows")

    test_targets: dict[int, np.ndarray] = {}
    for uid, g in df.groupby("uid", sort=False):
        test_targets[int(uid)] = np.unique(g["item_idx"].to_numpy(dtype=np.int64))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(test_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

    sizes = np.array([len(v) for v in test_targets.values()])
    print(
        f"[export] DONE in {time.time()-t0:.1f}s | users={len(test_targets):,} "
        f"| targets/user: mean={sizes.mean():.1f} median={np.median(sizes):.0f} "
        f"max={sizes.max()} | -> {OUT_PKL}  ({OUT_PKL.stat().st_size/2**20:.2f} MB)"
    )


if __name__ == "__main__":
    main()
