"""ID-based AGREE (Cao et al., SIGIR'18) — Phase 2, шаг 7.

Адаптация под наш сетап (frozen per-user scorer):

- Учим две ID-таблицы — `E_user[n_users+1, d]` и `E_item[n_items+1, d]`,
  обе с `padding_idx=0`. Это in-spirit оригинал: AGREE имеет user + item
  ID-эмбеддинги.
- Attention item-aware: alpha_{u,i} = softmax_u( h^T · tanh(W [E_u(u); E_i(i)]) ),
  softmax только по членам группы (с маскированием pad-членов).
- Group score: s_G(i) = sum_u alpha_{u,i} · s_{u,i}, где s_{u,i} приходит из
  замороженного скорера через `per_user_scores`. Это отличие от оригинала —
  там per-user score считался как <e_u, e_i>; у нас он внешний и не зависит
  от обучаемых ID-эмбеддингов.
- Item-эмбеддинги учатся через backprop по attention-логиту. Сравнение
  с Audio-AGREE (шаг 9) — чистая абляция: тот же attention, только источник
  эмбеддингов другой (audio вместо ID).

Remap raw uid → compact row делается внутри модуля через буфер
`_sorted_uids` + `torch.searchsorted`. Это даёт компактную таблицу размером
ровно `n_users+1` независимо от того, насколько разрежены raw uids в YAMBDA.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from .base import GroupAggregator


class IDBasedAGREE(GroupAggregator):
    """AGREE с item-aware attention на learnable ID-эмбеддингах."""

    def __init__(
        self,
        uid_list: Iterable[int],
        num_items: int,
        d_emb: int = 32,
        d_att: int | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        d_att = d_att or d_emb
        self.d_emb = int(d_emb)
        self.d_att = int(d_att)

        # Compact user remap: row 0 — PAD, rows 1..N — отсортированные raw uids.
        uids = sorted({int(u) for u in uid_list if int(u) != 0})
        sorted_with_pad = torch.tensor([0] + uids, dtype=torch.long)
        self.register_buffer("_sorted_uids", sorted_with_pad, persistent=True)
        self.num_users = len(uids)
        self.num_items = int(num_items)

        self.user_emb = nn.Embedding(self.num_users + 1, d_emb, padding_idx=0)
        self.item_emb = nn.Embedding(self.num_items + 1, d_emb, padding_idx=0)
        self.att_proj = nn.Linear(2 * d_emb, d_att, bias=True)
        self.att_out = nn.Linear(d_att, 1, bias=False)

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            self.user_emb.weight[0].zero_()
            self.item_emb.weight[0].zero_()
        nn.init.xavier_uniform_(self.att_proj.weight)
        nn.init.zeros_(self.att_proj.bias)
        nn.init.xavier_uniform_(self.att_out.weight)

    def _remap_uids(self, members: torch.Tensor) -> torch.Tensor:
        # searchsorted на CPU/GPU; outliers (uid не из train) уехали бы в out-of-range,
        # поэтому клипуем — для корректных данных no-op.
        rows = torch.searchsorted(self._sorted_uids, members.contiguous())
        return rows.clamp_(max=self.num_users)

    def forward(
        self,
        group_user_ids: torch.Tensor,  # [B, G]
        candidate_ids: torch.Tensor,  # [B, C]
        per_user_scores: torch.Tensor,  # [B, G, C]
        audio_embeds_items: torch.Tensor | None = None,
        audio_profiles_users: torch.Tensor | None = None,
        group_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del audio_embeds_items, audio_profiles_users  # ID-based: аудио игнорируем

        rows = self._remap_uids(group_user_ids)  # [B, G]
        eu = self.user_emb(rows)  # [B, G, d]
        ei = self.item_emb(candidate_ids)  # [B, C, d]

        B, G, d = eu.shape
        C = ei.shape[1]
        eu_exp = eu.unsqueeze(2).expand(B, G, C, d)  # [B, G, C, d]
        ei_exp = ei.unsqueeze(1).expand(B, G, C, d)
        pair = torch.cat([eu_exp, ei_exp], dim=-1)  # [B, G, C, 2d]
        logits = self.att_out(torch.tanh(self.att_proj(pair))).squeeze(-1)  # [B, G, C]

        if group_mask is not None:
            # softmax по G: pad-члены получают -inf и нулевой вклад.
            logits = logits.masked_fill(~group_mask.unsqueeze(-1), float("-inf"))

        alpha = torch.softmax(logits, dim=1)  # softmax по членам
        # Если у кандидата все члены — pad (теоретически невозможно при G≥1 реальных),
        # softmax(-inf) даёт NaN; зануляем для безопасности.
        alpha = torch.nan_to_num(alpha, nan=0.0)

        group_scores = (alpha * per_user_scores).sum(dim=1)  # [B, C]
        return group_scores
