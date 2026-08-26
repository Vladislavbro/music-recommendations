"""Bootstrap-CI над per-sample NDCG: маржинально, по размеру группы, попарно.

Все методы ресэмплятся по одной и той же сетке индексов — иначе paired-разности
несогласованы с маржинальными интервалами.

    per_sample = {"AudioAGREE": {10: arr, 20: arr}, ...}
    idx = bootstrap_indices(len(samples))
    summary_df = summarize(per_sample, idx, ks=[10, 20])
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "bootstrap_indices",
    "dump_per_sample",
    "latex_table",
    "load_per_sample",
    "paired_deltas",
    "summarize",
    "summarize_by_size",
]

PerSample = dict[str, dict[int, np.ndarray]]


def bootstrap_indices(n: int, n_boot: int = 1000, seed: int = 42) -> np.ndarray:
    """Матрица `[n_boot, n]` индексов ресэмплинга с возвращением."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _ci(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def summarize(per_sample: PerSample, idx: np.ndarray, ks: list[int]) -> pd.DataFrame:
    """Point estimate + 95% CI по всем группам, сортировка по первому K."""
    rows = []
    for method, by_k in per_sample.items():
        row: dict[str, object] = {"method": method}
        for k in ks:
            mean, lo, hi = _ci(np.asarray(by_k[k], dtype=np.float64), idx)
            row[f"NDCG@{k}"], row[f"NDCG@{k}_lo95"], row[f"NDCG@{k}_hi95"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"NDCG@{ks[0]}", ascending=False).reset_index(drop=True)


def summarize_by_size(
    per_sample: PerSample,
    sizes: np.ndarray,
    ks: list[int],
    n_boot: int = 1000,
    seed: int = 43,
) -> pd.DataFrame:
    """То же в разрезе размера группы; ресэмплинг — внутри каждого размера."""
    unique_sizes = sorted(np.unique(sizes).tolist())
    per_size_idx = _per_size_indices(sizes, unique_sizes, n_boot, seed)

    rows = []
    for method, by_k in per_sample.items():
        row: dict[str, object] = {"method": method}
        for k in ks:
            values = np.asarray(by_k[k], dtype=np.float64)
            for size in unique_sizes:
                mean, lo, hi = _ci(values[sizes == size], per_size_idx[size])
                col = f"NDCG@{k}[s={size}]"
                row[col], row[f"{col}_lo95"], row[f"{col}_hi95"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def _per_size_indices(
    sizes: np.ndarray, unique_sizes: list[int], n_boot: int, seed: int
) -> dict[int, np.ndarray]:
    """Индексы ресэмплинга локальные (0..n_size-1) — значения уже отфильтрованы по размеру."""
    rng = np.random.default_rng(seed)
    return {
        size: rng.integers(0, int((sizes == size).sum()), size=(n_boot, int((sizes == size).sum())))
        for size in unique_sizes
    }


def paired_deltas(
    per_sample: PerSample,
    pairs: list[tuple[str, str]],
    idx: np.ndarray,
    ks: list[int],
) -> pd.DataFrame:
    """Разности `mean(a) - mean(b)` на общей сетке ресэмплинга.

    `p_one_sided` — доля bootstrap-выборок с `Δ <= 0`, то есть H0 «a не лучше b».
    """
    rows = []
    for a, b in pairs:
        for k in ks:
            diff = np.asarray(per_sample[a][k], dtype=np.float64) - np.asarray(
                per_sample[b][k], dtype=np.float64
            )
            diff_boot = diff[idx].mean(axis=1)
            rows.append(
                {
                    "audio_method": a,
                    "id_method": b,
                    "K": k,
                    "delta_mean": float(diff.mean()),
                    "delta_lo95": float(np.quantile(diff_boot, 0.025)),
                    "delta_hi95": float(np.quantile(diff_boot, 0.975)),
                    "p_one_sided": float((diff_boot <= 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def latex_table(summary_df: pd.DataFrame, ks: list[int], digits: int = 4) -> str:
    """`summary_df` → tabular с колонками «значение / 95% CI» на каждый K."""
    header = (
        "\\begin{tabular}{l"
        + "rr" * len(ks)
        + "}\n\\toprule\nMethod"
        + "".join(f" & NDCG@{k} & 95\\% CI" for k in ks)
        + " \\\\\n\\midrule"
    )
    body = []
    for _, r in summary_df.iterrows():
        cells = [str(r["method"])]
        for k in ks:
            cells.append(f"{r[f'NDCG@{k}']:.{digits}f}")
            cells.append(f"[{r[f'NDCG@{k}_lo95']:.{digits}f},\\,{r[f'NDCG@{k}_hi95']:.{digits}f}]")
        body.append(" & ".join(cells) + " \\\\")
    return "\n".join([header, *body, "\\bottomrule\n\\end{tabular}"])


def dump_per_sample(
    per_sample: PerSample,
    sizes: np.ndarray,
    idx: np.ndarray,
    ks: list[int],
    path: str | Path,
) -> Path:
    """Сырые per-sample NDCG и сетка ресэмплинга — чтобы графики не пересчитывали eval."""
    dump = {
        f"{method}__NDCG@{k}": by_k[k].astype(np.float32)
        for method, by_k in per_sample.items()
        for k in ks
    }
    out = Path(path)
    np.savez_compressed(
        out, sizes=sizes.astype(np.int32), resample_idx=idx.astype(np.int32), **dump
    )
    return out


def load_per_sample(path: str | Path) -> tuple[PerSample, np.ndarray, np.ndarray]:
    """Обратная к `dump_per_sample`: `(per_sample, sizes, resample_idx)`."""
    data = np.load(Path(path))
    per_sample: PerSample = {}
    for key in data.files:
        if "__NDCG@" not in key:
            continue
        method, k = key.split("__NDCG@")
        per_sample.setdefault(method, {})[int(k)] = data[key].astype(np.float64)
    return per_sample, data["sizes"], data["resample_idx"]
