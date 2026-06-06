"""Club-Demo backend — онлайн групповые рекомендации поверх кэша SASRec.

Скорер заморожен и НЕ запускается здесь: его выход (top-200 на юзера) уже лежит
в `artifacts/user_scores_cache/scores.parquet`. Онлайн считается только агрегатор
(CPU, миллисекунды) на одну зонную группу (B=1) — точно по контракту Phase 2
(`src/aggregators/base.py`, см. CLAUDE.md §4).

Эндпоинт:
    POST /recommend  {user_ids: [int|str], method?: str, top_n?: int}
        -> {method, n_members, n_candidates, tracks: [{track_id, score}]}

Методы:
    audio_agree (default), group_cross_attn  — обучаемые audio-агрегаторы (best.pt);
    avg                                      — тривиальный mean(per_user_scores).

Запуск:
    uvicorn server:app --host 127.0.0.1 --port 8001
(см. Club-Demo/run_demo.sh)
"""
from __future__ import annotations

import sys
import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- project imports (src/ переиспользуем как есть) -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.group_trainer import (  # noqa: E402
    build_user_score_lookup,
    lookup_per_user_scores,
)
from src.aggregators.audio_agree import AudioAGREE  # noqa: E402
from src.aggregators.group_cross_attn import GroupCrossAttention  # noqa: E402
from src.eval.metrics import ndcg_from_ranking  # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
SCORES_PARQUET = ARTIFACTS / "user_scores_cache" / "scores.parquet"
EMBEDDINGS_NPY = ARTIFACTS / "audio" / "embeddings.npy"
PROFILES_NPY = ARTIFACTS / "audio" / "user_profiles.npy"
UID_TO_ROW_PKL = ARTIFACTS / "audio" / "uid_to_row.pkl"
TEST_TARGETS_PKL = ARTIFACTS / "test_targets" / "test_targets.pkl"
AGG_DIR = ARTIFACTS / "aggregators"

DEFAULT_METHOD = "audio_agree"
DEFAULT_TOP_N = 10
METRICS_K = 10                       # NDCG@K и coverage@K для live-панели
METRICS_METHODS = ("audio_agree", "avg")  # живой контраст H1: audio-aware vs тривиальный
# Ключи метрик, аккумулируемые в session running-average. ndcg/coverage — старые
# (group-level union-таргет); jain/disagreement — per-member→aggregate
# (канон grouprec: считаем satisfaction каждого члена, потом сворачиваем по группе).
METRIC_KEYS = ("ndcg", "coverage", "jain", "disagreement")
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Состояние, загружаемое один раз на старте
# ---------------------------------------------------------------------------
class Store:
    lookup: dict
    embeddings: np.ndarray            # [n_items, d_a]
    profiles: np.ndarray             # [n_users_rows, d_a]
    uid_to_row: dict                 # raw uid -> row в profiles
    d_audio: int
    aggregators: dict                # method -> nn.Module (audio)
    test_targets: dict               # raw uid -> np.ndarray[int64] held-out test-listens


STORE = Store()


class SessionMetrics:
    """Running-average по сессии для всех METRIC_KEYS (per-method).

    Накапливаем сумму и счётчик отдельно на каждый метод из METRICS_METHODS.
    Усредняем только по группам с непустым `targets` (метрики не определены иначе).
    Сбрасывается вместе со сбросом симуляции (`POST /reset_metrics`).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._acc = {
            m: {"n": 0, "sums": {k: 0.0 for k in METRIC_KEYS}} for m in METRICS_METHODS
        }

    def update(self, method: str, values: dict) -> None:
        a = self._acc[method]
        a["n"] += 1
        for k in METRIC_KEYS:
            a["sums"][k] += values[k]

    def snapshot(self) -> dict:
        out = {}
        for m, a in self._acc.items():
            n = a["n"]
            row = {"n": n}
            for k in METRIC_KEYS:
                row[k] = round(a["sums"][k] / n, 4) if n else None
            out[m] = row
        return out


METRICS = SessionMetrics()


def _load_aggregator(name: str, ctor) -> torch.nn.Module:
    ckpt = torch.load(AGG_DIR / name / "best.pt", map_location="cpu", weights_only=False)
    model = ctor()
    model.load_state_dict(ckpt["aggregator_state"])
    model.eval()
    model.to(DEVICE)
    return model


def load_store() -> None:
    t0 = time.time()
    print("[startup] loading scores cache ...", flush=True)
    scores_df = pd.read_parquet(SCORES_PARQUET, columns=["uid", "item_idx", "score"])
    STORE.lookup = build_user_score_lookup(scores_df)

    print("[startup] loading audio ...", flush=True)
    STORE.embeddings = np.load(EMBEDDINGS_NPY)
    STORE.profiles = np.load(PROFILES_NPY)
    with open(UID_TO_ROW_PKL, "rb") as f:
        STORE.uid_to_row = {int(k): int(v) for k, v in pickle.load(f).items()}
    STORE.d_audio = int(STORE.embeddings.shape[1])

    print("[startup] loading test targets (live metrics) ...", flush=True)
    if TEST_TARGETS_PKL.exists():
        with open(TEST_TARGETS_PKL, "rb") as f:
            STORE.test_targets = {
                int(k): np.asarray(v, dtype=np.int64) for k, v in pickle.load(f).items()
            }
    else:
        STORE.test_targets = {}
        print(
            f"[startup] WARNING: {TEST_TARGETS_PKL} не найден — live-метрики выключены. "
            "Сначала: python Club-Demo/backend/export_test_targets.py",
            flush=True,
        )

    print("[startup] loading aggregators ...", flush=True)
    STORE.aggregators = {
        "audio_agree": _load_aggregator("audio_agree", lambda: AudioAGREE(d_audio=STORE.d_audio)),
        "group_cross_attn": _load_aggregator(
            "group_cross_attn", lambda: GroupCrossAttention(d_audio=STORE.d_audio)
        ),
    }
    dt = time.time() - t0
    print(
        f"[startup] ready in {dt:.1f}s | users={len(STORE.lookup)} "
        f"items={STORE.embeddings.shape[0]} d_audio={STORE.d_audio} "
        f"test_target_users={len(STORE.test_targets)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Ядро: общие шаги для одной группы (B=1)
# ---------------------------------------------------------------------------
def _parse_members(user_ids: list[int]) -> list[int]:
    """Чистка raw uid: int-парсинг, дедуп с сохранением порядка, только те, что в кэше."""
    parsed = []
    for x in user_ids:
        try:
            parsed.append(int(x))
        except (TypeError, ValueError):
            continue
    return [u for u in dict.fromkeys(parsed) if u in STORE.lookup]


def _group_inputs(members: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """C_G = union(top-K членов) + per_user_scores [G, C] (item ∉ top-K → 0.0)."""
    cand_parts = [STORE.lookup[u][0] for u in members]
    candidates = np.unique(np.concatenate(cand_parts)).astype(np.int64)
    per_user = np.empty((len(members), candidates.shape[0]), dtype=np.float32)
    for gi, uid in enumerate(members):
        per_user[gi] = lookup_per_user_scores(STORE.lookup, uid, candidates, fill=0.0)
    return candidates, per_user


def _score(method: str, members: list[int], candidates: np.ndarray, per_user: np.ndarray) -> np.ndarray:
    """Групповой score per item для метода. `avg` — mean(per_user); иначе audio-агрегатор."""
    if method == "avg":
        return per_user.mean(axis=0)

    if method not in STORE.aggregators:
        method = DEFAULT_METHOD
    G, C = len(members), candidates.shape[0]
    item_audio = STORE.embeddings[candidates]                       # [C, d_a]
    user_audio = np.zeros((G, STORE.d_audio), dtype=np.float32)
    for gi, uid in enumerate(members):
        row = STORE.uid_to_row.get(uid)
        if row is not None:
            user_audio[gi] = STORE.profiles[row]

    with torch.no_grad():
        scores_t = STORE.aggregators[method](
            group_user_ids=torch.zeros((1, G), dtype=torch.long),
            candidate_ids=torch.from_numpy(candidates).unsqueeze(0),
            per_user_scores=torch.from_numpy(per_user).unsqueeze(0),
            audio_embeds_items=torch.from_numpy(item_audio).unsqueeze(0),
            audio_profiles_users=torch.from_numpy(user_audio).unsqueeze(0),
            group_mask=torch.ones((1, G), dtype=torch.bool),
            candidate_mask=torch.ones((1, C), dtype=torch.bool),
        )
    return scores_t.squeeze(0).numpy()


def recommend(user_ids: list[int], method: str, top_n: int) -> dict:
    method = (method or DEFAULT_METHOD).lower()
    members = _parse_members(user_ids)
    if not members:
        return {"method": method, "n_members": 0, "n_candidates": 0, "tracks": []}

    candidates, per_user = _group_inputs(members)
    if method not in ("avg",) and method not in STORE.aggregators:
        method = DEFAULT_METHOD
    scores = _score(method, members, candidates, per_user)

    C = candidates.shape[0]
    top_n = max(1, min(int(top_n), C))
    top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    tracks = [
        {"track_id": int(candidates[i]), "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]
    return {"method": method, "n_members": len(members), "n_candidates": C, "tracks": tracks}


# ---------------------------------------------------------------------------
# Live-метрики: NDCG@K + coverage@K на одну группу, по реальным test-listens
# ---------------------------------------------------------------------------
def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы top-k по убыванию score (argpartition + сортировка хвоста)."""
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _jain(xs: list[float]) -> float:
    """Jain's fairness index по per-member satisfaction: (Σx)² / (n·Σx²) ∈ (0,1].

    1.0 = всех обслужили одинаково (включая вырожденный all-zero — равная «нищета»),
    →1/n = всё ушло одному члену. Выше = справедливее.
    """
    s = float(sum(xs))
    s2 = float(sum(v * v for v in xs))
    if s2 <= 0.0:
        return 1.0
    return (s * s) / (len(xs) * s2)


def compute_group_metrics(members: list[int]) -> Optional[dict]:
    """NDCG@K и coverage@K для METRICS_METHODS на одной группе.

    Ground-truth: `targets = union(test_listens членов) ∩ C_G` — ровно контракт
    офлайн-eval (`src/eval/group_eval.py`). Если targets пуст → None (группа не
    учитывается в усреднении: NDCG не определён).

    coverage@K — доля членов группы, чей вкус «представлен» в топ-K: хотя бы один
    из top-K юзера попал в рекомендованные top-K. Иллюстрирует вырождение
    (рек выдаёт «фаворита одного гостя» → coverage низкий).

    Tier-1 (per-member→aggregate, канон grouprec). Сначала считаем NDCG@K каждого
    члена против ЕГО ЛИЧНЫХ test-listens (а не union), потом сворачиваем по группе:
        jain          — Jain's index по per-member NDCG (1 = равномерно, выше=честнее);
        disagreement  — std per-member NDCG (разброс сытости, ниже = лучше).
    Оцениваем только членов с непустым личным rel-set ∩ C_G (иначе satisfaction
    не определён). Группа доходит сюда ⇒ таких членов ≥ 1.

    Returns `{n_members, n_candidates, n_targets, n_eval_members,
    methods: {m: {ndcg, coverage, jain, disagreement}}}`
    или None, если метрику посчитать нельзя.
    """
    if not members:
        return None
    candidates, per_user = _group_inputs(members)

    # targets = union(test-listens) ∩ candidates (для group-level union-NDCG)
    tgt_parts = [
        STORE.test_targets[u] for u in members if u in STORE.test_targets and len(STORE.test_targets[u])
    ]
    if not tgt_parts:
        return None
    raw_targets = np.unique(np.concatenate(tgt_parts))
    targets = np.intersect1d(candidates, raw_targets, assume_unique=True)
    if targets.size == 0:
        return None
    rel_set = set(int(t) for t in targets)

    # Личные rel-set каждого члена (для per-member→aggregate). Члены без
    # достижимых личных таргетов из агрегации исключаются (satisfaction не определён).
    cand_set = set(int(c) for c in candidates)
    member_rel_sets: list[set] = []
    for uid in members:
        t = STORE.test_targets.get(uid)
        if t is None or len(t) == 0:
            continue
        rs = cand_set.intersection(int(i) for i in t)
        if rs:
            member_rel_sets.append(rs)

    out: dict = {
        "n_members": len(members),
        "n_candidates": int(candidates.shape[0]),
        "n_targets": int(targets.size),
        "n_eval_members": len(member_rel_sets),
        "methods": {},
    }
    for method in METRICS_METHODS:
        scores = _score(method, members, candidates, per_user)
        top_idx = _topk_indices(scores, METRICS_K)
        ranked_items = candidates[top_idx]
        ndcg = ndcg_from_ranking(ranked_items, rel_set, METRICS_K)

        rec_set = set(int(i) for i in ranked_items)
        represented = 0
        for uid in members:
            member_topk = STORE.lookup[uid][0]
            if rec_set.intersection(int(i) for i in member_topk):
                represented += 1
        coverage = represented / len(members)

        # per-member NDCG@K по тому же рек-рангу, но против личных таргетов члена
        pm = [ndcg_from_ranking(ranked_items, rs, METRICS_K) for rs in member_rel_sets]
        jain = _jain(pm)
        disagreement = float(np.std(pm)) if len(pm) > 1 else 0.0

        out["methods"][method] = {
            "ndcg": round(float(ndcg), 4),
            "coverage": round(float(coverage), 4),
            "jain": round(float(jain), 4),
            "disagreement": round(float(disagreement), 4),
        }
    return out


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Club-Demo backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_ids: list[int | str]
    method: Optional[str] = DEFAULT_METHOD
    top_n: Optional[int] = DEFAULT_TOP_N


class MetricsRequest(BaseModel):
    # Список зонных групп (каждая — список raw uid). Метрики считаются на каждой
    # группе и аккумулируются в running-average по сессии.
    groups: list[list[int | str]]


@app.on_event("startup")
def _startup() -> None:
    load_store()


@app.get("/health")
def health() -> dict:
    ready = hasattr(STORE, "lookup")
    return {
        "status": "ok" if ready else "loading",
        "users": len(STORE.lookup) if ready else 0,
        "methods": (["avg"] + list(STORE.aggregators.keys())) if ready else [],
        "metrics_enabled": bool(getattr(STORE, "test_targets", {})),
        "metrics_methods": list(METRICS_METHODS),
        "metrics_k": METRICS_K,
    }


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest) -> dict:
    return recommend(req.user_ids, req.method or DEFAULT_METHOD, req.top_n or DEFAULT_TOP_N)


@app.post("/metrics")
def metrics_endpoint(req: MetricsRequest) -> dict:
    """Считает NDCG@K + coverage@K по переданным группам, обновляет session running-average.

    Группы с пустым `targets` (нет достижимых test-listens) пропускаются. Ответ —
    `{k, methods, session, batch}`: `session` — скользящее среднее по всей сессии,
    `batch` — среднее по группам этого вызова (для дебага).
    """
    if not getattr(STORE, "test_targets", {}):
        return {"k": METRICS_K, "methods": list(METRICS_METHODS),
                "enabled": False, "session": METRICS.snapshot(), "batch": None}

    batch_acc = {m: {"n": 0, "sums": {k: 0.0 for k in METRIC_KEYS}} for m in METRICS_METHODS}
    n_groups_counted = 0
    for raw_group in req.groups:
        members = _parse_members(raw_group)
        res = compute_group_metrics(members)
        if res is None:
            continue
        n_groups_counted += 1
        for m, vals in res["methods"].items():
            METRICS.update(m, vals)
            b = batch_acc[m]
            b["n"] += 1
            for k in METRIC_KEYS:
                b["sums"][k] += vals[k]

    batch = {}
    for m, b in batch_acc.items():
        row = {"n": b["n"]}
        for k in METRIC_KEYS:
            row[k] = round(b["sums"][k] / b["n"], 4) if b["n"] else None
        batch[m] = row
    return {
        "k": METRICS_K,
        "methods": list(METRICS_METHODS),
        "enabled": True,
        "n_groups_counted": n_groups_counted,
        "session": METRICS.snapshot(),
        "batch": batch,
    }


@app.post("/reset_metrics")
def reset_metrics_endpoint() -> dict:
    """Сброс session running-average (вместе со сбросом симуляции на фронте)."""
    METRICS.reset()
    return {"status": "ok", "session": METRICS.snapshot()}
