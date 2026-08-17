"""Unit-тесты для `research/grouprec/aggregators/{base,agree,groupim}.py` (Phase 2, шаги 7–8).

Запуск: `pytest research/tests/test_aggregators.py -q`.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from grouprec.aggregators.agree import IDBasedAGREE
from grouprec.aggregators.audio_agree import AudioAGREE
from grouprec.aggregators.base import GroupAggregator
from grouprec.aggregators.group_cross_attn import GroupCrossAttention
from grouprec.aggregators.groupim import GroupIM
from grouprec.eval.group_eval import GroupSample
from grouprec.training.group_trainer import (
    GroupAggregatorTrainer,
    GroupTrainConfig,
    build_user_score_lookup,
    compute_pop_counts,
)

# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


def test_base_is_abstract():
    try:
        GroupAggregator()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("GroupAggregator must be abstract")


def test_agree_subclass_of_base():
    assert issubclass(IDBasedAGREE, GroupAggregator)


# ---------------------------------------------------------------------------
# IDBasedAGREE: remap
# ---------------------------------------------------------------------------


def test_remap_uids_pad_and_sparse():
    # Sparse uids — нарочно не от нуля и с дырами.
    uids = [3, 17, 42, 100]
    agg = IDBasedAGREE(uid_list=uids, num_items=10, d_emb=4)
    members = torch.tensor([[3, 17, 0], [100, 42, 0]], dtype=torch.long)
    rows = agg._remap_uids(members)
    # row 0 — PAD, дальше отсортированные uids: 3→1, 17→2, 42→3, 100→4
    expected = torch.tensor([[1, 2, 0], [4, 3, 0]], dtype=torch.long)
    assert torch.equal(rows, expected)


def test_remap_uids_zero_in_input_excluded():
    # Если 0 случайно попал в uid_list — он отбрасывается (зарезервирован под PAD).
    agg = IDBasedAGREE(uid_list=[0, 5, 7], num_items=10, d_emb=4)
    assert agg.num_users == 2
    rows = agg._remap_uids(torch.tensor([[5, 7, 0]], dtype=torch.long))
    assert rows.tolist() == [[1, 2, 0]]


# ---------------------------------------------------------------------------
# IDBasedAGREE: forward
# ---------------------------------------------------------------------------


def _make_aggregator(seed: int = 0, d: int = 8):
    torch.manual_seed(seed)
    return IDBasedAGREE(uid_list=[1, 2, 3, 4, 5], num_items=20, d_emb=d, d_att=d)


def test_forward_output_shape():
    agg = _make_aggregator()
    B, G, C = 3, 4, 7
    members = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0], [1, 4, 5, 2]], dtype=torch.long)
    cands = torch.randint(1, 21, (B, C), dtype=torch.long)
    scores = torch.randn(B, G, C)
    gmask = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True, True, True, True]]
    )
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()


def test_forward_pad_members_have_zero_attention():
    """Маска по G должна занулять веса pad-членов в softmax."""
    agg = _make_aggregator(seed=1)
    B, G, C = 1, 3, 5
    members = torch.tensor([[1, 2, 0]], dtype=torch.long)  # последний — pad
    cands = torch.tensor([[3, 7, 11, 13, 19]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    # Подменяем: для pad-члена кладём +1e6 в scores. Если маска работает,
    # это не должно влиять на group score.
    scores[0, 2, :] = 1e6
    gmask = torch.tensor([[True, True, False]])
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    # Если pad-член игнорируется, вклад 1e6 не должен прорваться.
    assert out.abs().max().item() < 100.0


def test_forward_attention_weights_sum_to_one_over_real_members():
    """Прямая проверка: alpha_{u,i} суммируется в 1 по реальным членам."""
    agg = _make_aggregator(seed=2)
    members = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7]], dtype=torch.long)
    gmask = torch.tensor([[True, True, True, False]])

    rows = agg._remap_uids(members)
    eu = agg.user_emb(rows)
    ei = agg.item_emb(cands)
    B, G, d = eu.shape
    C = ei.shape[1]
    eu_exp = eu.unsqueeze(2).expand(B, G, C, d)
    ei_exp = ei.unsqueeze(1).expand(B, G, C, d)
    pair = torch.cat([eu_exp, ei_exp], dim=-1)
    logits = agg.att_out(torch.tanh(agg.att_proj(pair))).squeeze(-1)
    logits = logits.masked_fill(~gmask.unsqueeze(-1), float("-inf"))
    alpha = torch.softmax(logits, dim=1)

    # По реальным членам сумма ≈ 1, по pad-члену ≈ 0.
    real_sum = alpha[0, :3, :].sum(dim=0)
    pad_sum = alpha[0, 3, :]
    assert torch.allclose(real_sum, torch.ones(C), atol=1e-5)
    assert torch.all(pad_sum == 0)


def test_forward_audio_args_ignored():
    """ID-based AGREE не должен падать при наличии аудио и не должен их использовать."""
    agg = _make_aggregator(seed=3)
    members = torch.tensor([[1, 2]], dtype=torch.long)
    cands = torch.tensor([[3, 7]], dtype=torch.long)
    scores = torch.randn(1, 2, 2)
    gmask = torch.ones(1, 2, dtype=torch.bool)
    cmask = torch.ones(1, 2, dtype=torch.bool)
    out_no_audio = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    out_with_audio = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=torch.randn(1, 2, 16),
        audio_profiles_users=torch.randn(1, 2, 16),
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.allclose(out_no_audio, out_with_audio)


def test_forward_backward_updates_parameters():
    agg = _make_aggregator(seed=4)
    members = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    scores = torch.randn(1, 3, 4)
    gmask = torch.ones(1, 3, dtype=torch.bool)
    cmask = torch.ones(1, 4, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    loss = out.sum()
    loss.backward()
    # Хотя бы у одного параметра должен быть ненулевой grad.
    has_grad = False
    for p in agg.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad


def test_forward_pad_member_value_irrelevant():
    """В pad-слоте значение uid не должно влиять на выход (gmask его маскирует)."""
    agg = _make_aggregator(seed=5)
    members_a = torch.tensor([[1, 2, 0]], dtype=torch.long)
    members_b = torch.tensor([[1, 2, 5]], dtype=torch.long)  # 5 — валидный uid, но в pad-слоте
    cands = torch.tensor([[3, 7, 11]], dtype=torch.long)
    scores = torch.randn(1, 3, 3)
    gmask = torch.tensor([[True, True, False]])
    cmask = torch.ones(1, 3, dtype=torch.bool)
    # Чтобы реально проверить инвариантность, нужно занулить pad-вход скоров —
    # иначе разные uids → разные эмбеддинги → разные attention-логиты, но pad всё
    # равно отсекается softmax-маской. Проверим равенство:
    out_a = agg(
        group_user_ids=members_a,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    out_b = agg(
        group_user_ids=members_b,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)


# ---------------------------------------------------------------------------
# end-to-end fit through GroupAggregatorTrainer
# ---------------------------------------------------------------------------


def _make_smoke_setup():
    rng = np.random.default_rng(123)
    n_items = 30
    uids = [10, 20, 30, 40, 50, 60]
    K = 8

    rows = []
    for uid in uids:
        items = rng.choice(np.arange(1, n_items + 1), size=K, replace=False)
        scores = rng.normal(5.0, 1.0, size=K).astype(np.float32)
        for rk, (it, sc) in enumerate(zip(items, scores, strict=True)):
            rows.append({"uid": uid, "item_idx": int(it), "score": float(sc), "rank": rk})
    scores_df = pd.DataFrame(rows)
    lookup = build_user_score_lookup(scores_df)

    pop_df = pd.DataFrame({"item_idx": rng.integers(1, n_items + 1, size=300)})
    pop_counts = compute_pop_counts(pop_df, n_items=n_items, smoothing=0.75)

    def _make_sample(members):
        cand_set = set()
        for u in members:
            items, _ = lookup[u]
            cand_set.update(items.tolist())
        cands = np.array(sorted(cand_set), dtype=np.int64)
        targets = rng.choice(cands, size=2, replace=False)
        targets = np.array(sorted(targets), dtype=np.int64)
        return GroupSample(members=tuple(members), candidates=cands, targets=targets)

    train = [_make_sample([10, 20]), _make_sample([20, 30, 40]), _make_sample([30, 40, 50, 60])]
    val = [_make_sample([10, 30]), _make_sample([20, 50, 60])]
    return {
        "lookup": lookup,
        "pop_counts": pop_counts,
        "uids": uids,
        "n_items": n_items,
        "train": train,
        "val": val,
    }


def test_agree_fit_smoke():
    tmp = Path(tempfile.mkdtemp(prefix="agree_test_"))
    try:
        s = _make_smoke_setup()
        cfg = GroupTrainConfig(
            n_epochs=2,
            batch_size=2,
            eval_batch_size=2,
            lr=1e-2,
            n_neg_per_pos=3,
            eval_k=(2, 4),
            early_stop_patience=10,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "agree_run"),
            device="cpu",
        )
        agg = IDBasedAGREE(uid_list=s["uids"], num_items=s["n_items"], d_emb=8)
        trainer = GroupAggregatorTrainer(
            aggregator=agg,
            cfg=cfg,
            user_score_lookup=s["lookup"],
            pop_counts=s["pop_counts"],
        )
        result = trainer.fit(s["train"], s["val"], verbose=False)
        primary_k = cfg.eval_k[0]
        key = f"best_val_NDCG@{primary_k}"
        assert key in result
        assert math.isfinite(result[key])
        assert 0.0 <= result[key] <= 1.0
        assert Path(result["checkpoint"]).exists()
        assert Path(result["metrics_csv"]).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_agree_predict_shapes():
    tmp = Path(tempfile.mkdtemp(prefix="agree_pred_"))
    try:
        s = _make_smoke_setup()
        cfg = GroupTrainConfig(
            n_epochs=1,
            batch_size=2,
            eval_batch_size=2,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "agree_run"),
            device="cpu",
        )
        agg = IDBasedAGREE(uid_list=s["uids"], num_items=s["n_items"], d_emb=8)
        trainer = GroupAggregatorTrainer(
            aggregator=agg,
            cfg=cfg,
            user_score_lookup=s["lookup"],
            pop_counts=s["pop_counts"],
        )
        scores = trainer.predict_group_scores(s["val"])
        assert len(scores) == len(s["val"])
        for sample, sc in zip(s["val"], scores, strict=True):
            assert sc.shape == (sample.n_candidates,)
            assert np.all(np.isfinite(sc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# GroupIM (шаг 8)
# ---------------------------------------------------------------------------


def _make_groupim(seed: int = 0, d: int = 8):
    torch.manual_seed(seed)
    return GroupIM(uid_list=[1, 2, 3, 4, 5], num_items=20, d_emb=d, d_att=d)


def test_groupim_subclass_of_base():
    assert issubclass(GroupIM, GroupAggregator)


def test_groupim_forward_output_shape():
    agg = _make_groupim()
    B, G, C = 3, 4, 7
    members = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0], [1, 4, 5, 2]], dtype=torch.long)
    cands = torch.randint(1, 21, (B, C), dtype=torch.long)
    scores = torch.randn(B, G, C)
    gmask = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True, True, True, True]]
    )
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()


def test_groupim_pad_members_have_zero_attention():
    """alpha pad-членов должна быть 0.

    Большие значения в их per_user_scores не должны прорываться.
    """
    agg = _make_groupim(seed=1)
    B, G, C = 1, 3, 5
    members = torch.tensor([[1, 2, 0]], dtype=torch.long)
    cands = torch.tensor([[3, 7, 11, 13, 19]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    scores[0, 2, :] = 1e6  # pad-член
    gmask = torch.tensor([[True, True, False]])
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.abs().max().item() < 100.0


def test_groupim_attention_is_item_agnostic():
    """В отличие от AGREE, alpha не зависит от item — все колонки `[B, G, C]` равны."""
    agg = _make_groupim(seed=2)
    members = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    gmask = torch.ones(1, 3, dtype=torch.bool)

    rows = agg._remap_uids(members)
    eu = agg.user_emb(rows)
    logits = agg.att_out(torch.tanh(agg.att_proj(eu))).squeeze(-1)
    logits = logits.masked_fill(~gmask, float("-inf"))
    alpha = torch.softmax(logits, dim=1)  # [B, G]
    # У alpha нет item-измерения — это и есть item-agnostic свойство.
    assert alpha.shape == (1, 3)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(1), atol=1e-5)

    # Проверим, что итоговый score = alpha @ per_user_scores: разные scores → разные outputs.
    scores_a = torch.zeros(1, 3, 4)
    scores_a[0, 0, :] = 1.0
    scores_b = torch.zeros(1, 3, 4)
    scores_b[0, 1, :] = 1.0
    out_a = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores_a,
        group_mask=gmask,
        candidate_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    out_b = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores_b,
        group_mask=gmask,
        candidate_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    # В обоих случаях все 4 кандидата получают одинаковый score (item-agnostic):
    assert torch.allclose(out_a[0], out_a[0, 0].expand(4), atol=1e-5)
    assert torch.allclose(out_b[0], out_b[0, 0].expand(4), atol=1e-5)


def test_groupim_audio_args_ignored():
    agg = _make_groupim(seed=3)
    members = torch.tensor([[1, 2]], dtype=torch.long)
    cands = torch.tensor([[3, 7]], dtype=torch.long)
    scores = torch.randn(1, 2, 2)
    gmask = torch.ones(1, 2, dtype=torch.bool)
    cmask = torch.ones(1, 2, dtype=torch.bool)
    out_no_audio = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    out_with_audio = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=torch.randn(1, 2, 16),
        audio_profiles_users=torch.randn(1, 2, 16),
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.allclose(out_no_audio, out_with_audio)


def test_groupim_mi_loss_finite_and_requires_grad():
    """MI-loss должен возвращать конечный скаляр с включённым autograd."""
    agg = _make_groupim(seed=4)
    agg.train()
    B, G, C = 4, 3, 5
    members = torch.tensor([[1, 2, 3], [2, 4, 0], [1, 5, 0], [3, 4, 5]], dtype=torch.long)
    cands = torch.randint(1, 21, (B, C), dtype=torch.long)
    scores = torch.randn(B, G, C)
    gmask = torch.tensor(
        [[True, True, True], [True, True, False], [True, True, False], [True, True, True]]
    )
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=torch.ones(B, C, dtype=torch.bool),
    )
    reg = agg.regularization_loss(batch={}, scores=out)
    assert reg.ndim == 0
    assert torch.isfinite(reg)
    assert reg.requires_grad
    reg.backward()
    # Проверим, что у mi_bilinear появился ненулевой grad.
    assert agg.mi_bilinear.grad is not None
    assert agg.mi_bilinear.grad.abs().sum().item() > 0


def test_groupim_mi_loss_zero_without_forward():
    """Без forward (или в eval) regularization_loss должен возвращать ноль без grad."""
    agg = _make_groupim(seed=5)
    agg.eval()
    out = torch.zeros(2, 3)
    reg = agg.regularization_loss(batch={}, scores=out)
    assert reg.ndim == 0
    assert float(reg) == 0.0


def test_groupim_mi_loss_reset_after_call():
    """Кеш должен очищаться после regularization_loss — повторный вызов возвращает 0."""
    agg = _make_groupim(seed=6)
    agg.train()
    B, G, C = 3, 2, 4
    members = torch.tensor([[1, 2], [3, 4], [2, 5]], dtype=torch.long)
    cands = torch.randint(1, 21, (B, C), dtype=torch.long)
    scores = torch.randn(B, G, C)
    gmask = torch.ones(B, G, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=torch.ones(B, C, dtype=torch.bool),
    )
    _ = agg.regularization_loss(batch={}, scores=out)
    reg2 = agg.regularization_loss(batch={}, scores=out)
    assert float(reg2) == 0.0


def test_groupim_mi_loss_small_batch_returns_zero():
    """При B<2 cross-batch негативы не определены — возвращаем 0."""
    agg = _make_groupim(seed=7)
    agg.train()
    members = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    scores = torch.randn(1, 3, 4)
    gmask = torch.ones(1, 3, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        group_mask=gmask,
        candidate_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    reg = agg.regularization_loss(batch={}, scores=out)
    assert float(reg) == 0.0


def test_groupim_fit_smoke_with_mi_loss():
    """End-to-end fit через trainer с включённым MI-регуляризатором."""
    tmp = Path(tempfile.mkdtemp(prefix="groupim_test_"))
    try:
        s = _make_smoke_setup()
        cfg = GroupTrainConfig(
            n_epochs=2,
            batch_size=2,
            eval_batch_size=2,
            lr=1e-2,
            n_neg_per_pos=3,
            eval_k=(2, 4),
            reg_loss_weight=0.5,  # включаем MI-loss
            early_stop_patience=10,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "groupim_run"),
            device="cpu",
        )
        agg = GroupIM(uid_list=s["uids"], num_items=s["n_items"], d_emb=8)
        trainer = GroupAggregatorTrainer(
            aggregator=agg,
            cfg=cfg,
            user_score_lookup=s["lookup"],
            pop_counts=s["pop_counts"],
        )
        result = trainer.fit(s["train"], s["val"], verbose=False)
        primary_k = cfg.eval_k[0]
        key = f"best_val_NDCG@{primary_k}"
        assert key in result
        assert math.isfinite(result[key])
        assert 0.0 <= result[key] <= 1.0
        assert Path(result["checkpoint"]).exists()
        assert Path(result["metrics_csv"]).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# AudioAGREE (шаг 9)
# ---------------------------------------------------------------------------


def _make_audio_agree(seed: int = 0, d_audio: int = 16, d_att: int = 8):
    torch.manual_seed(seed)
    return AudioAGREE(d_audio=d_audio, d_att=d_att)


def test_audio_agree_subclass_of_base():
    assert issubclass(AudioAGREE, GroupAggregator)


def test_audio_agree_forward_output_shape():
    agg = _make_audio_agree()
    B, G, C, d_a = 3, 4, 7, 16
    members = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0], [1, 4, 5, 2]], dtype=torch.long)
    cands = torch.randint(1, 21, (B, C), dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True, True, True, True]]
    )
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()


def test_audio_agree_raises_without_audio():
    agg = _make_audio_agree()
    members = torch.tensor([[1, 2]], dtype=torch.long)
    cands = torch.tensor([[3, 7]], dtype=torch.long)
    scores = torch.randn(1, 2, 2)
    gmask = torch.ones(1, 2, dtype=torch.bool)
    cmask = torch.ones(1, 2, dtype=torch.bool)
    try:
        _ = agg(
            group_user_ids=members,
            candidate_ids=cands,
            per_user_scores=scores,
            group_mask=gmask,
            candidate_mask=cmask,
        )
    except ValueError:
        return
    raise AssertionError("AudioAGREE must raise when audio inputs are missing")


def test_audio_agree_pad_members_have_zero_attention():
    """Маска по G зануляет веса pad-членов: large value в их per_user_scores не прорывается."""
    agg = _make_audio_agree(seed=1, d_audio=8, d_att=8)
    B, G, C, d_a = 1, 3, 5, 8
    members = torch.tensor([[1, 2, 0]], dtype=torch.long)
    cands = torch.tensor([[3, 7, 11, 13, 19]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    scores[0, 2, :] = 1e6
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.tensor([[True, True, False]])
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.abs().max().item() < 100.0


def test_audio_agree_attention_is_item_aware():
    """В отличие от GroupIM, alpha зависит от item — разные audio дают разные alpha по C."""
    agg = _make_audio_agree(seed=2, d_audio=8, d_att=8)
    B, G, C, d_a = 1, 2, 3, 8
    gmask = torch.tensor([[True, True]])
    # Явно различные item-аудио, чтобы alpha по C точно отличалась.
    item_aud = torch.zeros(B, C, d_a)
    item_aud[0, 0, 0] = 1.0
    item_aud[0, 1, 1] = 1.0
    item_aud[0, 2, 2] = 1.0
    user_aud = torch.randn(B, G, d_a)

    a_u_exp = user_aud.unsqueeze(2).expand(B, G, C, d_a)
    a_i_exp = item_aud.unsqueeze(1).expand(B, G, C, d_a)
    pair = torch.cat([a_u_exp, a_i_exp], dim=-1)
    logits = agg.phi_out(agg.act(agg.phi_hidden(pair))).squeeze(-1)
    logits = logits.masked_fill(~gmask.unsqueeze(-1), float("-inf"))
    alpha = torch.softmax(logits, dim=1)  # [B, G, C]
    diff = (alpha[0, :, 0] - alpha[0, :, 1]).abs().max().item()
    assert diff > 1e-4


def test_audio_agree_ignores_id_args():
    """AudioAGREE не должен опираться на group_user_ids и candidate_ids."""
    agg = _make_audio_agree(seed=3, d_audio=8, d_att=8)
    B, G, C, d_a = 1, 2, 3, 8
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)

    members_a = torch.tensor([[10, 20]], dtype=torch.long)
    members_b = torch.tensor([[999, 1234]], dtype=torch.long)
    cands_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands_b = torch.tensor([[777, 888, 999]], dtype=torch.long)
    out_a = agg(
        group_user_ids=members_a,
        candidate_ids=cands_a,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    out_b = agg(
        group_user_ids=members_b,
        candidate_ids=cands_b,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_audio_agree_missing_audio_does_not_nan():
    """Zero item-audio и user-profile (catalog/profile drift) дают конечные значения."""
    agg = _make_audio_agree(seed=4, d_audio=8, d_att=8)
    B, G, C, d_a = 1, 2, 4, 8
    members = torch.tensor([[1, 2]], dtype=torch.long)
    cands = torch.tensor([[3, 7, 11, 13]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    item_aud[0, 1] = 0.0  # «отсутствующий» аудио (catalog drift)
    user_aud = torch.randn(B, G, d_a)
    user_aud[0, 1] = 0.0  # «отсутствующий» профиль второго юзера
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.isfinite(out).all()


def test_audio_agree_backward_updates_parameters():
    agg = _make_audio_agree(seed=5, d_audio=8, d_att=8)
    B, G, C, d_a = 1, 3, 4, 8
    members = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in agg.parameters())
    assert has_grad


def test_audio_agree_fit_smoke():
    """End-to-end fit через trainer с фейковым item_audio и user_profiles."""
    tmp = Path(tempfile.mkdtemp(prefix="audio_agree_test_"))
    try:
        s = _make_smoke_setup()
        rng = np.random.default_rng(2026)
        d_a = 12
        item_audio = rng.normal(0.0, 1.0, size=(s["n_items"] + 1, d_a)).astype(np.float32)
        item_audio[0] = 0.0  # PAD
        for missing in [3, 11]:
            if 1 <= missing <= s["n_items"]:
                item_audio[missing] = 0.0  # эмулируем catalog drift
        user_profiles = rng.normal(0.0, 1.0, size=(len(s["uids"]), d_a)).astype(np.float32)
        uid_to_row = {uid: row for row, uid in enumerate(s["uids"])}

        cfg = GroupTrainConfig(
            n_epochs=2,
            batch_size=2,
            eval_batch_size=2,
            lr=1e-2,
            n_neg_per_pos=3,
            eval_k=(2, 4),
            early_stop_patience=10,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "audio_agree_run"),
            device="cpu",
        )
        agg = AudioAGREE(d_audio=d_a, d_att=8)
        trainer = GroupAggregatorTrainer(
            aggregator=agg,
            cfg=cfg,
            user_score_lookup=s["lookup"],
            pop_counts=s["pop_counts"],
            item_audio=item_audio,
            user_profiles=user_profiles,
            uid_to_row=uid_to_row,
        )
        result = trainer.fit(s["train"], s["val"], verbose=False)
        primary_k = cfg.eval_k[0]
        key = f"best_val_NDCG@{primary_k}"
        assert key in result
        assert math.isfinite(result[key])
        assert 0.0 <= result[key] <= 1.0
        assert Path(result["checkpoint"]).exists()
        assert Path(result["metrics_csv"]).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# GroupCrossAttention (шаг 10)
# ---------------------------------------------------------------------------


def _make_cross_attn(seed: int = 0, d_audio: int = 16, d_model: int = 16, n_heads: int = 4):
    torch.manual_seed(seed)
    return GroupCrossAttention(d_audio=d_audio, d_model=d_model, n_heads=n_heads)


def test_cross_attn_subclass_of_base():
    assert issubclass(GroupCrossAttention, GroupAggregator)


def test_cross_attn_invalid_head_config_raises():
    try:
        GroupCrossAttention(d_audio=16, d_model=10, n_heads=4)  # 10 % 4 != 0
    except ValueError:
        return
    raise AssertionError("d_model not divisible by n_heads must raise")


def test_cross_attn_forward_output_shape():
    agg = _make_cross_attn()
    B, G, C, d_a = 3, 4, 7, 16
    members = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0], [1, 4, 5, 2]], dtype=torch.long)
    cands = torch.randint(1, 21, (B, C), dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.tensor(
        [[True, True, True, False], [True, True, False, False], [True, True, True, True]]
    )
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()


def test_cross_attn_raises_without_audio():
    agg = _make_cross_attn()
    members = torch.tensor([[1, 2]], dtype=torch.long)
    cands = torch.tensor([[3, 7]], dtype=torch.long)
    scores = torch.randn(1, 2, 2)
    gmask = torch.ones(1, 2, dtype=torch.bool)
    cmask = torch.ones(1, 2, dtype=torch.bool)
    try:
        _ = agg(
            group_user_ids=members,
            candidate_ids=cands,
            per_user_scores=scores,
            group_mask=gmask,
            candidate_mask=cmask,
        )
    except ValueError:
        return
    raise AssertionError("GroupCrossAttention must raise when audio inputs are missing")


def test_cross_attn_pad_members_have_zero_attention():
    """Маска по G зануляет веса pad-членов: large value в их per_user_scores не прорывается."""
    agg = _make_cross_attn(seed=1, d_audio=8, d_model=8, n_heads=2)
    B, G, C, d_a = 1, 3, 5, 8
    members = torch.tensor([[1, 2, 0]], dtype=torch.long)
    cands = torch.tensor([[3, 7, 11, 13, 19]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    scores[0, 2, :] = 1e6
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.tensor([[True, True, False]])
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert out.abs().max().item() < 100.0


def test_cross_attn_attention_is_item_aware():
    """Разные item-аудио → разные alpha по C (item-aware свойство)."""
    agg = _make_cross_attn(seed=2, d_audio=8, d_model=8, n_heads=2)
    B, G, C, d_a = 1, 2, 3, 8
    gmask = torch.tensor([[True, True]])
    item_aud = torch.zeros(B, C, d_a)
    item_aud[0, 0, 0] = 1.0
    item_aud[0, 1, 1] = 1.0
    item_aud[0, 2, 2] = 1.0
    user_aud = torch.randn(B, G, d_a)

    H, dh = agg.n_heads, agg.d_head
    q = agg.q_proj(item_aud).view(B, C, H, dh).transpose(1, 2)
    k = agg.k_proj(user_aud).view(B, G, H, dh).transpose(1, 2)
    logits = torch.matmul(q, k.transpose(-1, -2)) * agg._scale
    logits = logits.masked_fill((~gmask).view(B, 1, 1, G), float("-inf"))
    alpha = torch.softmax(logits, dim=-1).mean(dim=1)  # [B, C, G]
    diff = (alpha[0, 0, :] - alpha[0, 1, :]).abs().max().item()
    assert diff > 1e-4


def test_cross_attn_ignores_id_args():
    """GroupCrossAttention не должен опираться на group_user_ids и candidate_ids."""
    agg = _make_cross_attn(seed=3, d_audio=8, d_model=8, n_heads=2)
    B, G, C, d_a = 1, 2, 3, 8
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)

    members_a = torch.tensor([[10, 20]], dtype=torch.long)
    members_b = torch.tensor([[999, 1234]], dtype=torch.long)
    cands_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands_b = torch.tensor([[777, 888, 999]], dtype=torch.long)
    out_a = agg(
        group_user_ids=members_a,
        candidate_ids=cands_a,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    out_b = agg(
        group_user_ids=members_b,
        candidate_ids=cands_b,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_cross_attn_missing_audio_does_not_nan():
    """Zero item-audio и user-profile дают конечные значения (catalog/profile drift)."""
    agg = _make_cross_attn(seed=4, d_audio=8, d_model=8, n_heads=2)
    B, G, C, d_a = 1, 2, 4, 8
    members = torch.tensor([[1, 2]], dtype=torch.long)
    cands = torch.tensor([[3, 7, 11, 13]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    item_aud[0, 1] = 0.0
    user_aud = torch.randn(B, G, d_a)
    user_aud[0, 1] = 0.0
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.isfinite(out).all()


def test_cross_attn_all_pad_no_nan():
    """Все члены — pad → alpha занулена через nan_to_num, выход = 0 без NaN."""
    agg = _make_cross_attn(seed=5, d_audio=8, d_model=8, n_heads=2)
    B, G, C, d_a = 1, 2, 3, 8
    members = torch.tensor([[0, 0]], dtype=torch.long)
    cands = torch.tensor([[1, 2, 3]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.zeros(B, G, dtype=torch.bool)  # все pad
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_cross_attn_backward_updates_parameters():
    agg = _make_cross_attn(seed=6, d_audio=8, d_model=8, n_heads=2)
    B, G, C, d_a = 1, 3, 4, 8
    members = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)
    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in agg.parameters())
    assert has_grad


def test_cross_attn_single_head_equivalence():
    """n_heads=1 — mean по головам тривиален; результат остаётся стандартным dot-product attn."""
    agg = _make_cross_attn(seed=7, d_audio=8, d_model=8, n_heads=1)
    B, G, C, d_a = 1, 3, 4, 8
    members = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cands = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    scores = torch.randn(B, G, C)
    item_aud = torch.randn(B, C, d_a)
    user_aud = torch.randn(B, G, d_a)
    gmask = torch.ones(B, G, dtype=torch.bool)
    cmask = torch.ones(B, C, dtype=torch.bool)

    out = agg(
        group_user_ids=members,
        candidate_ids=cands,
        per_user_scores=scores,
        audio_embeds_items=item_aud,
        audio_profiles_users=user_aud,
        group_mask=gmask,
        candidate_mask=cmask,
    )

    q = agg.q_proj(item_aud)  # [1, C, d_model]
    k = agg.k_proj(user_aud)  # [1, G, d_model]
    logits = torch.matmul(q, k.transpose(-1, -2)) * agg._scale  # [1, C, G]
    alpha = torch.softmax(logits, dim=-1)
    ref = (alpha * scores.transpose(1, 2)).sum(dim=-1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_cross_attn_fit_smoke():
    """End-to-end fit через trainer с фейковым item_audio и user_profiles."""
    tmp = Path(tempfile.mkdtemp(prefix="cross_attn_test_"))
    try:
        s = _make_smoke_setup()
        rng = np.random.default_rng(2026)
        d_a = 12
        item_audio = rng.normal(0.0, 1.0, size=(s["n_items"] + 1, d_a)).astype(np.float32)
        item_audio[0] = 0.0
        for missing in [3, 11]:
            if 1 <= missing <= s["n_items"]:
                item_audio[missing] = 0.0
        user_profiles = rng.normal(0.0, 1.0, size=(len(s["uids"]), d_a)).astype(np.float32)
        uid_to_row = {uid: row for row, uid in enumerate(s["uids"])}

        cfg = GroupTrainConfig(
            n_epochs=2,
            batch_size=2,
            eval_batch_size=2,
            lr=1e-2,
            n_neg_per_pos=3,
            eval_k=(2, 4),
            early_stop_patience=10,
            seed=0,
            log_every_steps=0,
            out_dir=str(tmp / "cross_attn_run"),
            device="cpu",
        )
        agg = GroupCrossAttention(d_audio=d_a, d_model=8, n_heads=2)
        trainer = GroupAggregatorTrainer(
            aggregator=agg,
            cfg=cfg,
            user_score_lookup=s["lookup"],
            pop_counts=s["pop_counts"],
            item_audio=item_audio,
            user_profiles=user_profiles,
            uid_to_row=uid_to_row,
        )
        result = trainer.fit(s["train"], s["val"], verbose=False)
        primary_k = cfg.eval_k[0]
        key = f"best_val_NDCG@{primary_k}"
        assert key in result
        assert math.isfinite(result[key])
        assert 0.0 <= result[key] <= 1.0
        assert Path(result["checkpoint"]).exists()
        assert Path(result["metrics_csv"]).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
