"""Abstract base class for group aggregators (Phase 2, шаг 7).

Контракт `forward` — расширение CLAUDE.md §4 с обязательными масками
(зафиксировано в phase_2_log, шаг 6):

```
group_user_ids:        LongTensor[B, G_max]                  # raw uids; pad-слот = 0
candidate_ids:         LongTensor[B, C_max]                  # compact item_idx; pad = 0
per_user_scores:       FloatTensor[B, G_max, C_max]          # 0 для item ∉ top-K и pad
audio_embeds_items:    FloatTensor[B, C_max, d_a] | None     # None для ID-based методов
audio_profiles_users:  FloatTensor[B, G_max, d_a] | None     # None для ID-based методов
group_mask:            BoolTensor[B, G_max]                  # True — реальный член
candidate_mask:        BoolTensor[B, C_max]                  # True — реальный кандидат
-> FloatTensor[B, C_max]                                     # групповой score per item
```

Любая реализация ОБЯЗАНА:
- Маскировать softmax/нормализацию по `group_mask` так, чтобы pad-члены не вносили
  вклад (иначе attention-веса смещены).
- Не полагаться на конкретные значения в pad-позициях `candidate_ids` — за маскирование
  pad-кандидатов в финальном ranking отвечает trainer (`-inf` после forward).
- Для audio-агрегаторов — также маскировать missing-audio items (см. phase_2_log, шаг 2,
  контракт по `audio_valid`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class GroupAggregator(nn.Module, ABC):
    """Базовый класс всех групповых агрегаторов Phase 2."""

    @abstractmethod
    def forward(
        self,
        group_user_ids: torch.LongTensor,
        candidate_ids: torch.LongTensor,
        per_user_scores: torch.Tensor,
        audio_embeds_items: Optional[torch.Tensor] = None,
        audio_profiles_users: Optional[torch.Tensor] = None,
        group_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Возвращает групповой score формы `[B, C_max]`."""
        raise NotImplementedError
