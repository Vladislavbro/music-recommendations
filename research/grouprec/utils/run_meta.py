"""`run.json` рядом с артефактами рана: метрики, версия кода, окружение.

write_run_json(out_dir, config=cfg, metrics={"NDCG@10": 0.09})
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["write_run_json"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_run_json(
    out_dir: str | Path,
    *,
    config: Any,
    metrics: dict[str, Any],
    name: str = "run.json",
) -> Path:
    """Пишет метрики рана вместе с git-ревизией и версиями библиотек."""
    out = Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_info(),
        "env": _env_info(),
        "config": asdict(config)
        if is_dataclass(config) and not isinstance(config, type)
        else config,
        "metrics": metrics,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    return out


def _git_info() -> dict[str, Any]:
    """`dirty` важнее самой ревизии: ран с незакоммиченными правками невоспроизводим."""
    revision = _git(["rev-parse", "HEAD"])
    if revision is None:
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": bool(_git(["status", "--porcelain"]))}


def _git(args: list[str]) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def _env_info() -> dict[str, Any]:
    versions = {}
    for mod in ("torch", "numpy", "pandas"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}
