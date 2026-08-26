"""Тесты bootstrap-обвязки: интервалы, срез по размеру, paired, round-trip дампа."""

from __future__ import annotations

import numpy as np
import pytest
from grouprec.eval.bootstrap import (
    bootstrap_indices,
    dump_per_sample,
    latex_table,
    load_per_sample,
    paired_deltas,
    summarize,
    summarize_by_size,
)


def _per_sample(n: int = 50) -> dict[str, dict[int, np.ndarray]]:
    rng = np.random.default_rng(0)
    return {
        "AudioAGREE": {10: rng.uniform(0.2, 0.4, n), 20: rng.uniform(0.3, 0.5, n)},
        "AGREE": {10: rng.uniform(0.0, 0.2, n), 20: rng.uniform(0.1, 0.3, n)},
    }


def test_indices_shape_and_seed():
    a = bootstrap_indices(50, n_boot=100, seed=42)
    assert a.shape == (100, 50)
    assert (a == bootstrap_indices(50, n_boot=100, seed=42)).all()
    assert a.max() < 50


def test_summary_point_estimate_is_plain_mean():
    per_sample = _per_sample()
    df = summarize(per_sample, bootstrap_indices(50, 200, 1), [10, 20]).set_index("method")
    for method, by_k in per_sample.items():
        assert df.loc[method, "NDCG@10"] == np.float64(by_k[10].mean())
        assert df.loc[method, "NDCG@10_lo95"] <= df.loc[method, "NDCG@10"]
        assert df.loc[method, "NDCG@10_hi95"] >= df.loc[method, "NDCG@10"]


def test_summary_sorted_by_first_k():
    df = summarize(_per_sample(), bootstrap_indices(50, 50, 1), [10, 20])
    assert df["method"].tolist() == ["AudioAGREE", "AGREE"]


def test_by_size_means_match_subset():
    per_sample = _per_sample()
    sizes = np.array([2] * 25 + [3] * 25)
    df = summarize_by_size(per_sample, sizes, [10], n_boot=50).set_index("method")
    expected = per_sample["AGREE"][10][sizes == 3].mean()
    assert df.loc["AGREE", "NDCG@10[s=3]"] == np.float64(expected)


def test_paired_delta_matches_difference():
    per_sample = _per_sample()
    df = paired_deltas(per_sample, [("AudioAGREE", "AGREE")], bootstrap_indices(50, 200, 1), [10])
    expected = per_sample["AudioAGREE"][10].mean() - per_sample["AGREE"][10].mean()
    assert df.loc[0, "delta_mean"] == pytest.approx(expected)
    assert 0.0 <= df.loc[0, "p_one_sided"] <= 1.0


def test_dump_load_round_trip(tmp_path):
    per_sample = _per_sample()
    sizes = np.full(50, 3, dtype=np.int64)
    idx = bootstrap_indices(50, 10, 1)
    path = dump_per_sample(per_sample, sizes, idx, [10, 20], tmp_path / "per_sample.npz")

    loaded, loaded_sizes, loaded_idx = load_per_sample(path)
    assert set(loaded) == set(per_sample)
    assert (loaded_sizes == sizes).all()
    assert (loaded_idx == idx).all()
    np.testing.assert_allclose(loaded["AGREE"][10], per_sample["AGREE"][10], rtol=1e-6)


def test_latex_table_has_row_per_method():
    df = summarize(_per_sample(), bootstrap_indices(50, 50, 1), [10])
    tex = latex_table(df, [10])
    rows = [line for line in tex.splitlines() if line.endswith(r"\\")]
    assert len(rows) == len(df) + 1  # + строка заголовка
    assert r"\begin{tabular}" in tex and "AudioAGREE" in tex
