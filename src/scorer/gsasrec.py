"""gSASRec architecture (Phase 1 per-user scorer).

Адаптация SASRec из `references/yambda/benchmarks/models/sasrec/model.py`
под наш интерфейс (см. CLAUDE.md, секция 4):

    forward(seq: [B, L])             -> hidden states [B, L, H]
    score(seq: [B, L], cand: [B, K]) -> scores [B, K]

Phase-1 решения (см. PHASE_1_LOG.md):
- Левый паддинг: `seq[:, -1]` всегда содержит самый свежий event.
  Поэтому `score()` берёт `hidden[:, -1, :]` без gather по длинам.
- Item indexing: 1..n_items, паддинг = 0.
- Position embedding: 0..L-1 (newest на позиции L-1).
- gBCE loss подключается снаружи (`gbce_loss.py`); сама модель loss
  не считает — это упрощает использование её же для скоринга.
"""

from __future__ import annotations

import torch
import torch.nn as nn


PAD_IDX = 0


class GSASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 200,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # +1 for padding idx 0; items are 1..n_items
        self.item_embeddings = nn.Embedding(
            num_embeddings=n_items + 1,
            embedding_dim=hidden_dim,
            padding_idx=PAD_IDX,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_dim,
        )

        self.layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation=nn.GELU(),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._init_weights(initializer_range)

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(param.data)
                else:
                    nn.init.trunc_normal_(
                        param.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            elif "bias" in name:
                nn.init.zeros_(param.data)
        # padding row stays zero
        with torch.no_grad():
            self.item_embeddings.weight[PAD_IDX].zero_()

    def forward(self, seq: torch.LongTensor) -> torch.Tensor:
        """seq: [B, L] (left-padded, items in 1..n_items, 0 = pad)
        returns: hidden states [B, L, H]
        """
        B, L = seq.shape
        assert L <= self.max_seq_len, f"seq len {L} > max_seq_len {self.max_seq_len}"

        item_emb = self.item_embeddings(seq)  # [B, L, H]

        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        pos_emb = self.position_embeddings(positions)  # [B, L, H]

        x = item_emb + pos_emb
        x = self.layernorm(x)
        x = self.dropout(x)

        pad_mask = seq == PAD_IDX  # [B, L], True where padded
        # zero-out padded positions before encoder (matches yambda reference)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Build per-batch 3D attention mask combining causal + pad-key masking.
        # Diagonal is always kept accessible: any query can attend to itself.
        # This avoids all-masked attention rows (PyTorch SDPA returns NaN for those)
        # for sequences with pad positions, without distribution shift for real positions
        # — pad keys remain blocked for any non-self query.
        causal = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )  # [L, L], True = future-masked
        key_pad = pad_mask.unsqueeze(1).expand(B, L, L)  # [B, L, L], pad in KEY dim
        diag = torch.eye(L, device=seq.device, dtype=torch.bool)  # [L, L]
        attn_mask = causal.unsqueeze(0) | (key_pad & ~diag.unsqueeze(0))  # [B, L, L]
        attn_mask = (
            attn_mask.unsqueeze(1)
            .expand(B, self.n_heads, L, L)
            .reshape(B * self.n_heads, L, L)
        )

        h = self.encoder(x, mask=attn_mask)
        return h

    def score(
        self, seq: torch.LongTensor, candidates: torch.LongTensor
    ) -> torch.Tensor:
        """seq: [B, L], candidates: [B, K] -> scores [B, K]

        Берём состояние последней (самой свежей) позиции и считаем
        dot-product с эмбеддингами кандидатов.
        """
        h = self.forward(seq)            # [B, L, H]
        last = h[:, -1, :]               # [B, H]  — newest, благодаря left-pad
        cand_emb = self.item_embeddings(candidates)  # [B, K, H]
        scores = torch.einsum("bh,bkh->bk", last, cand_emb)
        return scores
