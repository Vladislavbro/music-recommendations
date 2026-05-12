"""GroupIM (Sankar et al., SIGIR'20) — Phase 2, шаг 8.

Адаптация под наш сетап (frozen per-user scorer + ephemeral groups):

- Learnable `E_user[n_users+1, d]` (`padding_idx=0`); item-эмбеддинги не нужны
  (per-user scores приходят из замороженного скорера, attention item-agnostic).
- Item-agnostic attention: `alpha_u = softmax_u(h^T · tanh(W · e_u))`. Это чистый
  контраст к AGREE из шага 7, где attention был item-aware. В оригинальной
  GroupIM (Sankar et al.) attention тоже item-agnostic, поэтому отступление
  от paper'а минимальное — мы лишь подменяем источник `p_{u,i}` (там MLP-выход
  user-encoder'а, у нас — кэш SASRec).
- Group score: `s_G(i) = sum_u alpha_u · s_{u,i}` (та же `alpha` для всех items,
  в отличие от AGREE).
- Group representation `h_G = sum_u alpha_u · e_u` — нужно для MI-loss.
- **MI-loss** (бинарный дискриминатор). Bilinear `D(e_u, h_G) = e_u^T W h_G`.
  Положительные пары — `(e_u, h_G)` для `u ∈ G`, негативы — `(e_u, h_{G'})`,
  где `G'` — другая группа того же батча (cross-batch shuffle). Loss —
  стандартный BCE через `logsigmoid` (численно стабильно). Pad-члены
  исключаются маской.

Контракт `regularization_loss(batch, scores)` для `GroupAggregatorTrainer`:
после forward модуль кеширует `(member_emb, group_repr, group_mask)`, а на
вызове reg-loss собирает MI без повторного прохода. Кеш сбрасывается после
каждого использования. В eval-режиме `regularization_loss` возвращает 0
(скор предсказание идёт без MI). Альтернативный публичный API — `mi_loss(...)`
с явными аргументами — оставлен для unit-тестов.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import GroupAggregator


class GroupIM(GroupAggregator):
    """GroupIM с item-agnostic attention и MI-регуляризацией для ephemeral групп."""

    def __init__(
        self,
        uid_list: Iterable[int],
        num_items: int,
        d_emb: int = 32,
        d_att: Optional[int] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        d_att = d_att or d_emb
        self.d_emb = int(d_emb)
        self.d_att = int(d_att)

        # Compact user remap (как в AGREE): row 0 — PAD, далее отсортированные raw uids.
        uids = sorted({int(u) for u in uid_list if int(u) != 0})
        self.register_buffer(
            "_sorted_uids", torch.tensor([0] + uids, dtype=torch.long), persistent=True
        )
        self.num_users = len(uids)
        self.num_items = int(num_items)

        self.user_emb = nn.Embedding(self.num_users + 1, d_emb, padding_idx=0)
        self.att_proj = nn.Linear(d_emb, d_att, bias=True)
        self.att_out = nn.Linear(d_att, 1, bias=False)
        # Bilinear MI-дискриминатор: D(e_u, h_G) = e_u^T W h_G.
        self.mi_bilinear = nn.Parameter(torch.empty(d_emb, d_emb))

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            self.user_emb.weight[0].zero_()
        nn.init.xavier_uniform_(self.att_proj.weight)
        nn.init.zeros_(self.att_proj.bias)
        nn.init.xavier_uniform_(self.att_out.weight)
        nn.init.xavier_uniform_(self.mi_bilinear)

        # Кеш для regularization_loss (заполняется в forward в train-режиме).
        self._cached_member_emb: Optional[torch.Tensor] = None
        self._cached_group_repr: Optional[torch.Tensor] = None
        self._cached_group_mask: Optional[torch.Tensor] = None

    def _remap_uids(self, members: torch.Tensor) -> torch.Tensor:
        rows = torch.searchsorted(self._sorted_uids, members.contiguous())
        return rows.clamp_(max=self.num_users)

    def forward(
        self,
        group_user_ids: torch.Tensor,            # [B, G]
        candidate_ids: torch.Tensor,             # [B, C]
        per_user_scores: torch.Tensor,           # [B, G, C]
        audio_embeds_items: Optional[torch.Tensor] = None,
        audio_profiles_users: Optional[torch.Tensor] = None,
        group_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del candidate_ids, audio_embeds_items, audio_profiles_users, candidate_mask

        rows = self._remap_uids(group_user_ids)               # [B, G]
        eu = self.user_emb(rows)                              # [B, G, d]

        # Item-agnostic attention по членам.
        att_logits = self.att_out(torch.tanh(self.att_proj(eu))).squeeze(-1)  # [B, G]
        if group_mask is not None:
            att_logits = att_logits.masked_fill(~group_mask, float("-inf"))
        alpha = torch.softmax(att_logits, dim=1)              # [B, G]
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Group representation для MI-loss.
        group_repr = (alpha.unsqueeze(-1) * eu).sum(dim=1)    # [B, d]

        # Кешируем только в train-режиме, чтобы не таскать state в инференсе.
        if self.training:
            self._cached_member_emb = eu
            self._cached_group_repr = group_repr
            self._cached_group_mask = group_mask
        else:
            self._cached_member_emb = None
            self._cached_group_repr = None
            self._cached_group_mask = None

        # Group score: alpha [B, G] (item-agnostic) × per_user_scores [B, G, C].
        group_scores = (alpha.unsqueeze(-1) * per_user_scores).sum(dim=1)  # [B, C]
        return group_scores

    # ------------------------------------------------------------------
    # MI-loss
    # ------------------------------------------------------------------

    def mi_loss(
        self,
        member_emb: torch.Tensor,         # [B, G, d]
        group_repr: torch.Tensor,         # [B, d]
        group_mask: Optional[torch.Tensor] = None,  # [B, G], True — реальный член
        neg_group_repr: Optional[torch.Tensor] = None,  # [B, d]; по умолчанию — batch-shift
    ) -> torch.Tensor:
        """Бинарный MI-loss. Возвращает скаляр.

        Если негативы не переданы, использует cross-batch shuffle (`torch.roll(., 1)`).
        В этом случае при `B < 2` MI-loss не определён и возвращается 0 (без grad).
        """
        B, G, d = member_emb.shape

        if neg_group_repr is None:
            if B < 2:
                return member_emb.new_zeros(())
            perm = torch.roll(torch.arange(B, device=member_emb.device), shifts=1)
            neg_group_repr = group_repr[perm]

        # eu_proj[b, g] = e_u^T W  ∈ R^d
        eu_proj = torch.matmul(member_emb, self.mi_bilinear)  # [B, G, d]
        pos_logits = (eu_proj * group_repr.unsqueeze(1)).sum(dim=-1)     # [B, G]
        neg_logits = (eu_proj * neg_group_repr.unsqueeze(1)).sum(dim=-1) # [B, G]

        # BCE: -[log σ(pos) + log(1 - σ(neg))].
        per_member = -(F.logsigmoid(pos_logits) + F.logsigmoid(-neg_logits))  # [B, G]

        if group_mask is not None:
            mask = group_mask.to(per_member.dtype)
        else:
            mask = torch.ones_like(per_member)
        denom = mask.sum().clamp_min(1.0)
        return (per_member * mask).sum() / denom

    def regularization_loss(
        self,
        batch: dict,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Хук для `GroupAggregatorTrainer`. Использует кеш из последнего forward.

        Если кеш пуст (forward не выполнялся в train-режиме или batch size < 2 для
        cross-batch негативов), возвращает скалярный 0.
        """
        del batch
        if (
            self._cached_member_emb is None
            or self._cached_group_repr is None
        ):
            return scores.new_zeros(())
        loss = self.mi_loss(
            member_emb=self._cached_member_emb,
            group_repr=self._cached_group_repr,
            group_mask=self._cached_group_mask,
        )
        # Сбрасываем кеш — следующий forward переполнит.
        self._cached_member_emb = None
        self._cached_group_repr = None
        self._cached_group_mask = None
        return loss
