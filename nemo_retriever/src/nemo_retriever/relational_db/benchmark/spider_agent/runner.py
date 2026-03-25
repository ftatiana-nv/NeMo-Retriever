"""Locate the official Spider2 repo and invoke ``methods/spider-agent-lite/run.py``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def get_spider2_repo_root() -> Path:
    """Path to a local clone of ``xlang-ai/Spider2`` (must contain ``methods/spider-agent-lite``)."""
    raw = os.environ.get("SPIDER2_REPO_ROOT") or os.environ.get("SPIDER2_ROOT")
    if not raw or not str(raw).strip():
        raise RuntimeError(
            "Set SPIDER2_REPO_ROOT to the root of a cloned Spider2 repository "
            "(https://github.com/xlang-ai/Spider2), e.g. export SPIDER2_REPO_ROOT=$HOME/Spider2"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SPIDER2_REPO_ROOT is not a directory: {root}")
    return root


def spider_agent_lite_dir(repo_root: Path | None = None) -> Path:
    d = (repo_root or get_spider2_repo_root()) / "methods" / "spider-agent-lite"
    if not d.is_dir():
        raise FileNotFoundError(
            f"Expected spider-agent-lite at {d}. "
            "Use a full Spider2 clone on the default branch."
        )
    return d


def run_py_path(repo_root: Path | None = None) -> Path:
    p = spider_agent_lite_dir(repo_root) / "run.py"
    if not p.is_file():
        raise FileNotFoundError(f"Missing upstream run.py: {p}")
    return p


def run_spider_agent_lite(
    *,
    test_path: Path,
    model: str,
    suffix: str,
    output_dir: Path | None = None,
    local_only: bool = True,
    example_index: str = "all",
    example_name: str = "",
    extra_args: list[str] | None = None,
    repo_root: Path | None = None,
) -> int:
    """Run upstream ``run.py`` with ``cwd`` = ``spider-agent-lite``. Returns process exit code.

    Official Spider-Agent-Lite expects Docker and credentials for BigQuery/Snowflake; use
    ``--local_only`` for SQLite/DuckDB-style ``local*`` tasks only.

    Parameters
    ----------
    test_path:
        Path to a ``.jsonl`` task file (e.g. spider2-lite.jsonl). Rows should include
        ``instance_id``; upstream filters on ``id`` for ``--example_name``, so prefer
        JSONL produced by :func:`prepare_task_jsonl_for_upstream`.
    """
    lite = spider_agent_lite_dir(repo_root)
    run_py = run_py_path(repo_root)
    out = output_dir or (lite / "output")
    out.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        str(run_py),
        "--model",
        model,
        "-s",
        suffix,
        "--test_path",
        str(test_path.resolve()),
        "--example_index",
        example_index,
        "--output_dir",
        str(out.resolve()),
    ]
    if example_name:
        cmd.extend(["--example_name", example_name])
    if local_only:
        cmd.append("--local_only")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=str(lite),
        env=env,
    )
    return int(proc.returncode)
