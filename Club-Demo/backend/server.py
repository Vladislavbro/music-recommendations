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

ARTIFACTS = PROJECT_ROOT / "artifacts"
SCORES_PARQUET = ARTIFACTS / "user_scores_cache" / "scores.parquet"
EMBEDDINGS_NPY = ARTIFACTS / "audio" / "embeddings.npy"
PROFILES_NPY = ARTIFACTS / "audio" / "user_profiles.npy"
UID_TO_ROW_PKL = ARTIFACTS / "audio" / "uid_to_row.pkl"
AGG_DIR = ARTIFACTS / "aggregators"

DEFAULT_METHOD = "audio_agree"
DEFAULT_TOP_N = 10
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


STORE = Store()


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
        f"items={STORE.embeddings.shape[0]} d_audio={STORE.d_audio}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Ядро: рекомендации на одну группу (B=1)
# ---------------------------------------------------------------------------
def recommend(user_ids: list[int], method: str, top_n: int) -> dict:
    method = (method or DEFAULT_METHOD).lower()

    parsed = []
    for x in user_ids:
        try:
            parsed.append(int(x))
        except (TypeError, ValueError):
            continue
    members = [u for u in dict.fromkeys(parsed) if u in STORE.lookup]
    if not members:
        return {"method": method, "n_members": 0, "n_candidates": 0, "tracks": []}

    # C_G = union(top-K членов)
    cand_parts = [STORE.lookup[u][0] for u in members]
    candidates = np.unique(np.concatenate(cand_parts)).astype(np.int64)
    G, C = len(members), len(candidates)

    # per_user_scores [G, C]; item ∉ top-K юзера → 0.0
    per_user = np.empty((G, C), dtype=np.float32)
    for gi, uid in enumerate(members):
        per_user[gi] = lookup_per_user_scores(STORE.lookup, uid, candidates, fill=0.0)

    if method == "avg":
        scores = per_user.mean(axis=0)
    else:
        if method not in STORE.aggregators:
            method = DEFAULT_METHOD
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
        scores = scores_t.squeeze(0).numpy()

    top_n = max(1, min(int(top_n), C))
    top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    tracks = [
        {"track_id": int(candidates[i]), "score": round(float(scores[i]), 4)}
        for i in top_idx
    ]
    return {"method": method, "n_members": G, "n_candidates": C, "tracks": tracks}


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
    }


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest) -> dict:
    return recommend(req.user_ids, req.method or DEFAULT_METHOD, req.top_n or DEFAULT_TOP_N)
