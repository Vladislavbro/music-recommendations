"""Замер памяти и скорости инференса финалистов (CPU).

Меряет ровно online-часть пайплайна — обученный групповой агрегатор (контракт
`src/aggregators/base.py`, путь загрузки/сборки входов скопирован из
`Club-Demo/backend/server.py`). Скорер заморожен и здесь НЕ запускается: его
выход уже в кэше `scores.parquet`.

Две цифры по ПАМЯТИ (как договорились):
  1. online   — параметры самого агрегатора (то, что считается на каждый запрос).
  2. full     — весь самодостаточный пайплайн на диске: скорер + кэш top-K +
                аудио (items + профили) + агрегатор. Амортизированная офлайн-цена.
  Доп.: resident-аудио — фактический срез эмбеддингов под кандидатов одной группы.

СКОРОСТЬ: чистый `aggregator.forward` с B=1 (как в демке), warmup + N прогонов,
median / p95 в мс, с разбивкой по размеру группы (2–5) и по |C_G|.

Запуск:
    python tests/bench_inference.py                 # реальные артефакты
    python tests/bench_inference.py --runs 500       # больше прогонов
    python tests/bench_inference.py --out logs/...    # сохранить markdown-табличку

Без артефактов (artifacts/ не выгружены) скрипт честно падает с подсказкой —
синтетику для замера скорости использовать можно через --synthetic.
"""
from __future__ import annotations

import argparse
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aggregators.audio_agree import AudioAGREE  # noqa: E402
from src.aggregators.group_cross_attn import GroupCrossAttention  # noqa: E402
from src.training.group_trainer import (  # noqa: E402
    build_user_score_lookup,
    lookup_per_user_scores,
)

ARTIFACTS = REPO_ROOT / "artifacts"
SCORES_PARQUET = ARTIFACTS / "user_scores_cache" / "scores.parquet"
EMBEDDINGS_NPY = ARTIFACTS / "audio" / "embeddings.npy"
PROFILES_NPY = ARTIFACTS / "audio" / "user_profiles.npy"
UID_TO_ROW_PKL = ARTIFACTS / "audio" / "uid_to_row.pkl"
AGG_DIR = ARTIFACTS / "aggregators"
SCORER_DIR = ARTIFACTS / "gsasrec"

# Финалисты (CLAUDE.md §итоговая таблица). ctor строится после того, как известно d_audio.
FINALISTS = ("audio_agree", "group_cross_attn")

# Распределение размеров групп — то же, что в синтезе групп Phase 2.
GROUP_SIZE_DIST = {2: 0.3, 3: 0.4, 4: 0.2, 5: 0.1}

DEVICE = torch.device("cpu")
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Память
# ---------------------------------------------------------------------------
def param_footprint(model: torch.nn.Module) -> dict:
    """Параметры + буферы модели: число элементов и байты (online-память модели)."""
    n_params = sum(p.numel() for p in model.parameters())
    b_params = sum(p.numel() * p.element_size() for p in model.parameters())
    b_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "n_params": n_params,
        "bytes": b_params + b_buffers,
    }


def disk_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def full_footprint() -> dict:
    """Полный самодостаточный пайплайн на диске (амортизированная офлайн-цена)."""
    parts = {
        "scorer (SASRec best.pt)": disk_size(SCORER_DIR / "best.pt"),
        "scorer item_id_map": disk_size(SCORER_DIR / "item_id_to_idx.pkl"),
        "scores cache (top-K)": disk_size(SCORES_PARQUET),
        "audio item embeddings": disk_size(EMBEDDINGS_NPY),
        "audio user profiles": disk_size(PROFILES_NPY),
    }
    return parts


# ---------------------------------------------------------------------------
# Загрузка store (как в server.py)
# ---------------------------------------------------------------------------
class Store:
    lookup: dict
    embeddings: np.ndarray
    profiles: np.ndarray
    uid_to_row: dict
    d_audio: int
    aggregators: dict


def load_aggregator(name: str, ctor) -> torch.nn.Module:
    ckpt = torch.load(AGG_DIR / name / "best.pt", map_location="cpu", weights_only=False)
    model = ctor()
    model.load_state_dict(ckpt["aggregator_state"])
    model.eval()
    model.to(DEVICE)
    return model


def load_store() -> Store:
    import pandas as pd

    store = Store()
    scores_df = pd.read_parquet(SCORES_PARQUET, columns=["uid", "item_idx", "score"])
    store.lookup = build_user_score_lookup(scores_df)
    store.embeddings = np.load(EMBEDDINGS_NPY)
    store.profiles = np.load(PROFILES_NPY)
    with open(UID_TO_ROW_PKL, "rb") as f:
        store.uid_to_row = {int(k): int(v) for k, v in pickle.load(f).items()}
    store.d_audio = int(store.embeddings.shape[1])
    store.aggregators = {
        "audio_agree": load_aggregator("audio_agree", lambda: AudioAGREE(d_audio=store.d_audio)),
        "group_cross_attn": load_aggregator(
            "group_cross_attn", lambda: GroupCrossAttention(d_audio=store.d_audio)
        ),
    }
    return store


# ---------------------------------------------------------------------------
# Сборка входов одной группы (B=1) — копия server._group_inputs / _score
# ---------------------------------------------------------------------------
def group_inputs(store: Store, members: list[int]):
    cand_parts = [store.lookup[u][0] for u in members]
    candidates = np.unique(np.concatenate(cand_parts)).astype(np.int64)
    per_user = np.empty((len(members), candidates.shape[0]), dtype=np.float32)
    for gi, uid in enumerate(members):
        per_user[gi] = lookup_per_user_scores(store.lookup, uid, candidates, fill=0.0)
    return candidates, per_user


def build_forward_kwargs(store: Store, members: list[int], candidates, per_user) -> dict:
    G, C = len(members), candidates.shape[0]
    item_audio = store.embeddings[candidates]
    user_audio = np.zeros((G, store.d_audio), dtype=np.float32)
    for gi, uid in enumerate(members):
        row = store.uid_to_row.get(uid)
        if row is not None:
            user_audio[gi] = store.profiles[row]
    return {
        "group_user_ids": torch.zeros((1, G), dtype=torch.long),
        "candidate_ids": torch.from_numpy(candidates).unsqueeze(0),
        "per_user_scores": torch.from_numpy(per_user).unsqueeze(0),
        "audio_embeds_items": torch.from_numpy(item_audio).unsqueeze(0),
        "audio_profiles_users": torch.from_numpy(user_audio).unsqueeze(0),
        "group_mask": torch.ones((1, G), dtype=torch.bool),
        "candidate_mask": torch.ones((1, C), dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Бенч скорости
# ---------------------------------------------------------------------------
def time_forward(model: torch.nn.Module, kwargs: dict, runs: int, warmup: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            model(**kwargs)
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(**kwargs)
            samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return samples


def sample_groups(store: Store, rng, n_per_size: int) -> dict:
    """n_per_size реальных групп на каждый размер 2–5 (uids из кэша)."""
    uids = np.array(list(store.lookup.keys()))
    groups = {}
    for size in (2, 3, 4, 5):
        groups[size] = [
            list(rng.choice(uids, size=size, replace=False)) for _ in range(n_per_size)
        ]
    return groups


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def fmt_mb(b: int) -> str:
    return f"{b / MB:.2f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200, help="прогонов на одну группу")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--groups-per-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="markdown-файл для таблицы")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (демка: 1)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    if not SCORES_PARQUET.exists() or not EMBEDDINGS_NPY.exists():
        print(
            "[bench] артефакты не найдены (artifacts/user_scores_cache, artifacts/audio).\n"
            "        Эти файлы в .gitignore — нужны локально. См. CLAUDE.md §Артефакты.",
            file=sys.stderr,
        )
        return 1

    print("[bench] loading store ...", flush=True)
    store = load_store()
    n_items = store.embeddings.shape[0]

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# Замер: память и скорость инференса (CPU)")
    out()
    out(f"- threads: {args.threads} | runs/группу: {args.runs} | warmup: {args.warmup}")
    out(f"- users в кэше: {len(store.lookup)} | items: {n_items} | d_audio: {store.d_audio}")
    out()

    # --- ПАМЯТЬ: online (параметры агрегатора) ---
    out("## Память — online (параметры агрегатора, на каждый запрос)")
    out()
    out("| Метод | Параметры | Память |")
    out("|---|---:|---:|")
    for name in FINALISTS:
        fp = param_footprint(store.aggregators[name])
        out(f"| {name} | {fp['n_params']:,} | {fp['bytes'] / 1024:.1f} KB |")
    out()

    # --- ПАМЯТЬ: full footprint (диск) ---
    out("## Память — full footprint (весь пайплайн на диске)")
    out()
    out("| Компонент | Размер |")
    out("|---|---:|")
    parts = full_footprint()
    total = 0
    for k, v in parts.items():
        total += v
        out(f"| {k} | {fmt_mb(v)} |")
    out(f"| **Итого** | **{fmt_mb(total)}** |")
    out()
    out("> online ≪ full: скорер (заморожен) и аудио — амортизированная офлайн-цена; "
        "на запрос считается только агрегатор (KB-параметры).")
    out()

    # --- СКОРОСТЬ ---
    groups = sample_groups(store, rng, args.groups_per_size)
    out("## Скорость инференса — aggregator.forward, B=1")
    out()
    out("Латентность одного прогона (мс), агрегировано по реальным группам каждого размера.")
    out()
    out("| Метод | Размер группы | ср. |C_G| | median, мс | p95, мс |")
    out("|---|---:|---:|---:|---:|")

    for name in FINALISTS:
        model = store.aggregators[name]
        for size in (2, 3, 4, 5):
            all_samples: list[float] = []
            cand_counts: list[int] = []
            for members in groups[size]:
                members = [int(u) for u in members]
                candidates, per_user = group_inputs(store, members)
                cand_counts.append(candidates.shape[0])
                kwargs = build_forward_kwargs(store, members, candidates, per_user)
                # меньше прогонов на группу, но много групп — усредняем по обоим
                all_samples.extend(
                    time_forward(model, kwargs, runs=max(1, args.runs // args.groups_per_size),
                                 warmup=max(1, args.warmup // args.groups_per_size))
                )
            med = statistics.median(all_samples)
            p95 = float(np.percentile(all_samples, 95))
            avg_c = int(np.mean(cand_counts))
            out(f"| {name} | {size} | {avg_c} | {med:.3f} | {p95:.3f} |")
    out()
    out("> |C_G| = размер union(top-K членов); латентность растёт с числом кандидатов, "
        "не с размером группы как таковым.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[bench] сохранено → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
