"""gBCE loss (Petrov & Macdonald, RecSys'23).

Перенос из `references/gSASRec-pytorch/train_gsasrec.py` (lines 59-72) в
самостоятельную функцию по сигнатуре из CLAUDE.md.

Идея: при sampled-softmax/BCE с малым числом негативов модель смещается
к переоценке вероятностей позитива. Калибровочный коэффициент
    beta = alpha * ((1 - 1/alpha) * t + 1/alpha),  alpha = n_neg / (n_items - 1)
трансформирует положительный логит так, что при `t=1` восстанавливается
полный softmax, при `t=0` — обычный BCE (= оригинальный SASRec).

Численная стабильность: pos_logits/neg_logits приводятся к float64
(как в авторском коде), внутри — clamp с eps. На выходе loss возвращается
во float32 для совместимости с amp/optimizer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gbce_loss(
    pos_logits: torch.Tensor,  # [B] либо [..., 1] — логиты позитивов
    neg_logits: torch.Tensor,  # [..., n_neg]      — логиты негативов
    n_items: int,
    n_neg: int,
    t: float = 0.75,
) -> torch.Tensor:
    """Возвращает скаляр (mean BCE по всем элементам и слотам кандидатов).

    Формы pos/neg должны быть broadcast-совместимы по leading dims, например:
      - pos [B],         neg [B, n_neg]
      - pos [B, L],      neg [B, L, n_neg]   — для seq2seq-обучения
      - pos [B, L, 1],   neg [B, L, n_neg]
    """
    if pos_logits.shape[-1:] != (1,):
        pos_logits = pos_logits.unsqueeze(-1)  # -> [..., 1]

    assert neg_logits.shape[-1] == n_neg, (
        f"neg_logits last dim {neg_logits.shape[-1]} != n_neg {n_neg}"
    )

    # alpha — доля сэмплированных негативов от всего каталога (минус позитив).
    alpha = n_neg / max(n_items - 1, 1)
    # beta из формулы Петрова & Макдональда.
    beta = alpha * ((1.0 - 1.0 / alpha) * t + 1.0 / alpha)

    eps = 1e-10
    pos64 = pos_logits.to(torch.float64)
    neg64 = neg_logits.to(torch.float64)

    pos_probs = torch.clamp(torch.sigmoid(pos64), eps, 1.0 - eps)
    pos_probs_adj = torch.clamp(pos_probs.pow(-beta), 1.0 + eps, torch.finfo(torch.float64).max)
    to_log = torch.clamp(1.0 / (pos_probs_adj - 1.0), eps, torch.finfo(torch.float64).max)
    pos_logits_transformed = to_log.log()  # [..., 1]

    logits = torch.cat([pos_logits_transformed, neg64], dim=-1)  # [..., 1+n_neg]
    targets = torch.zeros_like(logits)
    targets[..., 0] = 1.0

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    return loss.to(pos_logits.dtype if pos_logits.is_floating_point() else torch.float32)
