"""Pairwise BPR loss для обучения групповых агрегаторов (Phase 2, шаг 6).

L = -mean( log_sigmoid(pos - neg) ),
где `pos` — скор положительного item'а, `neg` — скор негатива.
Broadcasting: shape пары `pos_scores - neg_scores` определяется через
обычные torch-правила, поэтому одинаково работает для:
- `pos_scores [B]`, `neg_scores [B]` — 1:1 pair-mining;
- `pos_scores [B, 1]`, `neg_scores [B, K]` — 1:K с усреднением по K.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise BPR loss `-log sigmoid(pos - neg)`.

    Args:
        pos_scores: скор позитивов. Shape `[B]` или `[B, 1]`.
        neg_scores: скор негативов. Shape `[B]` или `[B, K]`. Должен бродкаститься
            с `pos_scores`.
        eps: численный пол для `log_sigmoid` (используется stable-реализация PyTorch,
            но `eps` оставлен для совместимости и явного контроля).

    Returns:
        Скаляр-loss (mean по всем парам).
    """
    if pos_scores.dim() == 1 and neg_scores.dim() == 2:
        pos_scores = pos_scores.unsqueeze(1)
    diff = pos_scores - neg_scores
    # log_sigmoid численно стабилен у torch (использует softplus(-x)).
    loss = -F.logsigmoid(diff)
    return loss.mean()
