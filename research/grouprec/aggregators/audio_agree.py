"""Audio-AGREE — Phase 2, шаг 9.

Прямой аналог `IDBasedAGREE`, в котором обучаемые ID-эмбеддинги
заменены на аудиоэмбеддинги YAMBDA:

- `a_i ∈ R^{d_a}` — item-аудио из `artifacts/audio/embeddings.npy`.
- `a_bar_u ∈ R^{d_a}` — user audio profile (mean по valid items истории,
  см. шаг 4: `artifacts/audio/user_profiles.npy`).
- `phi`: 2-слойный MLP с GELU. На вход — `concat(a_bar_u, a_i)`.
- Attention item-aware: `alpha_{u,i} = softmax_u( phi([a_bar_u; a_i]) )`.
- Group score: `s_G(i) = Σ_u alpha_{u,i} · s_{u,i}`, где `s_{u,i}` — кэш
  замороженного SASRec (`per_user_scores`).

Никаких обучаемых параметров на стороне юзеров и items — только `phi`.
Это и есть «всё на аудио» из CLAUDE.md §4: разница с AGREE — только
источник эмбеддингов; attention-форма и BPR-обвязка те же.

Контракты по маскам и audio-валидности (phase_2_log, шаги 2 и 6):
- `group_mask` — pad-члены маскируются в softmax по G (получают -inf).
- Items без аудио (`||a_i|| == 0`) и юзеры без профиля (`||a_bar_u|| == 0`)
  специально не маскируются — `phi` через ReLU/GELU-нелинейности
  естественно даёт логит, опирающийся на оставшийся ненулевой вход.
  Это совместимо с контрактом «нулевой вклад в логит» из шага 2:
  отсутствующий компонент не влияет на mlp-проекцию своей половины.
- Pad-кандидатов маскирует Trainer (`-inf` после forward).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import GroupAggregator


class AudioAGREE(GroupAggregator):
    """Audio-AGREE с item-aware attention на аудиоэмбеддингах."""

    def __init__(
        self,
        d_audio: int = 128,
        d_att: int = 64,
    ) -> None:
        super().__init__()
        self.d_audio = int(d_audio)
        self.d_att = int(d_att)

        self.phi_hidden = nn.Linear(2 * self.d_audio, self.d_att, bias=True)
        self.phi_out = nn.Linear(self.d_att, 1, bias=False)
        self.act = nn.GELU()

        nn.init.xavier_uniform_(self.phi_hidden.weight)
        nn.init.zeros_(self.phi_hidden.bias)
        nn.init.xavier_uniform_(self.phi_out.weight)

    def forward(
        self,
        group_user_ids: torch.Tensor,  # [B, G] — игнорируется (аудио уже агрегировано)
        candidate_ids: torch.Tensor,  # [B, C] — игнорируется
        per_user_scores: torch.Tensor,  # [B, G, C]
        audio_embeds_items: torch.Tensor | None = None,  # [B, C, d_a]
        audio_profiles_users: torch.Tensor | None = None,  # [B, G, d_a]
        group_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del group_user_ids, candidate_ids, candidate_mask

        if audio_embeds_items is None or audio_profiles_users is None:
            raise ValueError("AudioAGREE requires `audio_embeds_items` and `audio_profiles_users`")

        a_u = audio_profiles_users  # [B, G, d_a]
        a_i = audio_embeds_items  # [B, C, d_a]

        B, G, d_a = a_u.shape
        C = a_i.shape[1]
        if d_a != self.d_audio:
            raise ValueError(f"d_audio mismatch: got {d_a}, expected {self.d_audio}")

        a_u_exp = a_u.unsqueeze(2).expand(B, G, C, d_a)
        a_i_exp = a_i.unsqueeze(1).expand(B, G, C, d_a)
        pair = torch.cat([a_u_exp, a_i_exp], dim=-1)  # [B, G, C, 2*d_a]
        logits = self.phi_out(self.act(self.phi_hidden(pair))).squeeze(-1)  # [B, G, C]

        if group_mask is not None:
            logits = logits.masked_fill(~group_mask.unsqueeze(-1), float("-inf"))

        alpha = torch.softmax(logits, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        group_scores = (alpha * per_user_scores).sum(dim=1)  # [B, C]
        return group_scores
