"""Run upstream Spider-Agent-Lite and harvest ``generated_sql/spider_agent/*.sql``.

This does **not** reimplement Spider-Agent; it invokes the official
``methods/spider-agent-lite/run.py`` from a cloned `Spider2`_ repository.

.. _Spider2: https://github.com/xlang-ai/Spider2
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_retriever.relational_db.benchmark.spider_agent.extract_sql import (
    harvest_sql_from_spider_output,
)
from nemo_retriever.relational_db.benchmark.spider_agent.runner import (
    get_spider2_repo_root,
    run_spider_agent_lite,
    spider_agent_lite_dir,
)


def prepare_task_jsonl_for_upstream(
    source: Path,
    *,
    instance_id_prefix: str | None = None,
) -> Path:
    """Write a temp JSONL with ``id`` set from ``instance_id`` (upstream ``run.py`` filters on ``id``)."""
    lines_out: list[str] = []
    with open(source, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("instance_id", "")
            if instance_id_prefix and not str(iid).startswith(instance_id_prefix):
                continue
            if "id" not in row:
                row["id"] = iid
            lines_out.append(json.dumps(row, ensure_ascii=False))

    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", text=True)
    os.close(fd)
    p = Path(tmp_path)
    p.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return p


def run_spider_agent_benchmark(
    *,
    jsonl_path: Path,
    output_dir: Path,
    instance_id_prefix: str | None = None,
    model: str | None = None,
    suffix: str | None = None,
    local_only: bool | None = None,
) -> None:
    """Clone-based workflow: run official ``run.py`` then harvest ``.sql`` into ``output_dir``.

    Environment
    -----------
    SPIDER2_REPO_ROOT
        Required. Root of ``xlang-ai/Spider2`` clone.
    SPIDER_AGENT_MODEL / LLM_MODEL / OPENAI_MODEL
        Passed to ``--model`` (default ``gpt-4o``).
    SPIDER_AGENT_SUFFIX
        Experiment suffix ``-s`` (default ``nemo-benchmark``).
    SPIDER_AGENT_OUTPUT_DIR
        Optional. Override upstream output directory (default: ``<spider-agent-lite>/output``).
    SPIDER_AGENT_LOCAL_ONLY
        If ``0``/``false``, do not pass ``--local_only`` (BigQuery/Snowflake/DBT tasks need creds + Docker).
    """
    root = get_spider2_repo_root()
    lite = spider_agent_lite_dir(root)

    model = (
        model
        or os.environ.get("SPIDER_AGENT_MODEL")
        or os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o"
    )
    suffix = suffix or os.environ.get("SPIDER_AGENT_SUFFIX", "nemo-benchmark")

    if local_only is None:
        v = os.environ.get("SPIDER_AGENT_LOCAL_ONLY", "1").strip().lower()
        local_only = v not in ("0", "false", "no", "off")

    out_upstream = (
        Path(os.environ["SPIDER_AGENT_OUTPUT_DIR"]).expanduser().resolve()
        if os.environ.get("SPIDER_AGENT_OUTPUT_DIR")
        else (lite / "output")
    )

    tmp = prepare_task_jsonl_for_upstream(
        jsonl_path, instance_id_prefix=instance_id_prefix
    )
    try:
        code = run_spider_agent_lite(
            test_path=tmp,
            model=model,
            suffix=suffix,
            output_dir=out_upstream,
            local_only=local_only,
            example_index="all",
        )
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    if code != 0:
        raise RuntimeError(
            f"Spider-Agent-Lite run.py exited with code {code}. "
            "Check Docker, credentials, and methods/spider-agent-lite logs."
        )

    written, skipped = harvest_sql_from_spider_output(
        out_upstream,
        output_dir,
        instance_id_prefix=instance_id_prefix,
        skip_existing=True,
    )
    print(
        f"[spider-agent] Harvest: written={written}, skipped (existing)={skipped} -> {output_dir}"
    )


def run_spider_agent_single(
    question: str,
    *,
    instance_id: str,
    db: str | None = None,
    model: str | None = None,
    suffix: str | None = None,
) -> str:
    """Run upstream ``run.py`` on a one-line JSONL (for ``debug_retriever_sql`` single mode)."""
    root = get_spider2_repo_root()
    lite = spider_agent_lite_dir(root)
    model = (
        model
        or os.environ.get("SPIDER_AGENT_MODEL")
        or os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o"
    )
    suffix = suffix or os.environ.get("SPIDER_AGENT_SUFFIX", "nemo-single")

    row: dict = {
        "instance_id": instance_id,
        "id": instance_id,
        "question": question,
    }
    if db is not None:
        row["db"] = db

    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", text=True)
    os.close(fd)
    p = Path(tmp_path)
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    out_upstream = (
        Path(os.environ["SPIDER_AGENT_OUTPUT_DIR"]).expanduser().resolve()
        if os.environ.get("SPIDER_AGENT_OUTPUT_DIR")
        else (lite / "output")
    )

    v = os.environ.get("SPIDER_AGENT_LOCAL_ONLY", "1").strip().lower()
    local_only = v not in ("0", "false", "no", "off")

    try:
        code = run_spider_agent_lite(
            test_path=p,
            model=model,
            suffix=suffix,
            output_dir=out_upstream,
            local_only=local_only,
            example_index="all",
        )
    finally:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    if code != 0:
        raise RuntimeError(f"Spider-Agent-Lite run.py exited with code {code}")

    with TemporaryDirectory(prefix="spider_harvest_") as td:
        d = Path(td)
        harvest_sql_from_spider_output(
            out_upstream,
            d,
            instance_id_prefix=instance_id,
            skip_existing=False,
        )
        one = d / f"{instance_id}.sql"
        if one.is_file():
            return one.read_text(encoding="utf-8").strip()
    return ""
