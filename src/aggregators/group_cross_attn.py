"""Group cross-attention — Phase 2, шаг 10.

Multi-head cross-attention поверх аудиоэмбеддингов. Естественное
multi-head обобщение `AudioAGREE`: вместо MLP `phi(concat(a_u, a_i))`
для одного скалярного логита используем `H` голов scaled dot-product
attention с раздельными линейными проекциями Q и K.

- `Q = W_Q · a_i ∈ R^{d_model}` — запросы по кандидатам.
- `K = W_K · a_bar_u ∈ R^{d_model}` — ключи по членам группы.
- Логиты `[B, H, C, G] = Q · K^⊤ / sqrt(d_head)`.
- Softmax по G (с маскированием pad-членов через `group_mask`).
- Attention-веса усредняются по головам: `alpha[B, C, G] = mean_h(...)`.
- Group score сохраняет общий контракт всех агрегаторов Phase 2:
  `s_G(i) = Σ_u alpha[u, i] · s_{u,i}`, где `s_{u,i}` — `per_user_scores`
  из замороженного SASRec.

Заметки по дизайну:
- V-проекция не нужна: мы используем attention-веса для линейной
  комбинации `per_user_scores` (скаляров), а не d_model-векторов.
  Это сохраняет «frozen scorer + аггрегатор только взвешивает» —
  общий контракт всех 4 методов Phase 2.
- Усреднение по головам коммутирует с умножением на `per_user_scores`
  (линейность), поэтому эквивалентно «голова считает свой `s_G`, и мы
  усредняем». Так что для интерпретации головы — это разные «взгляды»
  на одного и того же кандидата.
- Catalog drift по аудио (нулевые `a_i` / `a_bar_u`) специально не
  маскируется: нулевой ключ/запрос даёт логит ≈ 0, и через softmax по G
  даёт почти-равномерное распределение — мягкий fallback на AVG по группе.

Контракты по маскам (phase_2_log, шаги 2 и 6):
- `group_mask` — pad-члены маскируются в softmax по G (-inf).
- `audio_valid` по items явно не используется — см. заметку выше.
- Pad-кандидатов маскирует Trainer (`-inf` после forward).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import GroupAggregator


class GroupCrossAttention(GroupAggregator):
    """Multi-head cross-attention на аудио, Q = items, K = group members."""

    def __init__(
        self,
        d_audio: int = 128,
        d_model: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_audio = int(d_audio)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self._scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(self.d_audio, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_audio, self.d_model, bias=True)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

    def forward(
        self,
        group_user_ids: torch.Tensor,            # [B, G] — игнорируется
        candidate_ids: torch.Tensor,             # [B, C] — игнорируется
        per_user_scores: torch.Tensor,           # [B, G, C]
        audio_embeds_items: Optional[torch.Tensor] = None,    # [B, C, d_a]
        audio_profiles_users: Optional[torch.Tensor] = None,  # [B, G, d_a]
        group_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del group_user_ids, candidate_ids, candidate_mask

        if audio_embeds_items is None or audio_profiles_users is None:
            raise ValueError(
                "GroupCrossAttention requires `audio_embeds_items` and `audio_profiles_users`"
            )

        a_i = audio_embeds_items     # [B, C, d_a]
        a_u = audio_profiles_users   # [B, G, d_a]

        B, G, d_a = a_u.shape
        C = a_i.shape[1]
        if d_a != self.d_audio:
            raise ValueError(f"d_audio mismatch: got {d_a}, expected {self.d_audio}")

        H, dh = self.n_heads, self.d_head

        q = self.q_proj(a_i).view(B, C, H, dh).transpose(1, 2)  # [B, H, C, dh]
        k = self.k_proj(a_u).view(B, G, H, dh).transpose(1, 2)  # [B, H, G, dh]

        logits = torch.matmul(q, k.transpose(-1, -2)) * self._scale  # [B, H, C, G]

        if group_mask is not None:
            key_pad = (~group_mask).view(B, 1, 1, G)                # [B, 1, 1, G]
            logits = logits.masked_fill(key_pad, float("-inf"))

        alpha = torch.softmax(logits, dim=-1)                       # [B, H, C, G]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = alpha.mean(dim=1)                                   # [B, C, G]

        scores_T = per_user_scores.transpose(1, 2)                  # [B, C, G]
        group_scores = (alpha * scores_T).sum(dim=-1)               # [B, C]
        return group_scores
