#!/usr/bin/env python3
"""Debug script: retrieve from LanceDB (structured table), then answer questions with SQL.

Modes (symmetrical layout under ``benchmark/``):

- **sql-tool** (default): LanceDB + RAG LLM → ``benchmark/sql_tool/generated_sql/sql_tool/``
- **deep-agent**: Deep Agent (``benchmark/deep_agent/generate_sql.py``) → ``benchmark/deep_agent/generated_sql/deep_agent/``
- **spider-agent**: upstream ``Spider2/methods/spider-agent-lite`` (requires ``SPIDER2_REPO_ROOT`` + Docker) → ``benchmark/spider_agent/generated_sql/spider_agent/``

Single question (default, sql-tool):
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py

Deep-agent batch:
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch --mode deep-agent

Loop over benchmark/spider2-lite.jsonl (shared by sql-tool and deep-agent batch):
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch

Run only instances whose instance_id starts with a prefix (e.g. "local"):
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch --prefix local

Batch **resume:** rows are skipped if output already exists (deep-agent: answer JSON under
``benchmark/deep_agent/generated_answers/deep_agent/``; sql-tool: ``.sql`` in the batch output dir).

Requires: LanceDB + LLM env for sql-tool; ``./spider2.duckdb`` (or ``DUCKDB_PATH``) + LLM for deep-agent batch.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Paths relative to repo root (run from repo root)
REPO_ROOT = Path(__file__).resolve().parent
_BENCHMARK = (
    REPO_ROOT
    / "nemo_retriever"
    / "src"
    / "nemo_retriever"
    / "relational_db"
    / "benchmark"
)
SQL_TOOL_DIR = _BENCHMARK / "sql_tool"
DEEP_AGENT_DIR = _BENCHMARK / "deep_agent"
SPIDER_AGENT_DIR = _BENCHMARK / "spider_agent"

# Shared Spider2-lite task list (all models / pipelines); not under sql_tool only.
SPIDER2_LITE_JSONL = _BENCHMARK / "spider2-lite.jsonl"

# Subfolder under ``generated_sql/`` so batch outputs are clearly separated by pipeline.
_OUTPUT_MODE_DIR = {
    "deep-agent": "deep_agent",
    "sql-tool": "sql_tool",
    "spider-agent": "spider_agent",
}


def _canonical_batch_output_dir(mode: str) -> Path:
    """Default directory for batch ``.sql`` files: ``generated_sql/<...>/`` per mode."""
    sub = _OUTPUT_MODE_DIR.get(mode, "sql_tool")
    if mode == "deep-agent":
        return (DEEP_AGENT_DIR / "generated_sql" / sub).resolve()
    if mode == "spider-agent":
        return (SPIDER_AGENT_DIR / "generated_sql" / sub).resolve()
    return (SQL_TOOL_DIR / "generated_sql" / sub).resolve()


def _safe_filename(instance_id: str) -> str:
    """Make instance_id safe for a filename (no path separators or reserved chars)."""
    return re.sub(r'[^\w\-.]', "_", instance_id) or "unknown"


def _deep_agent_generated_answers_dir() -> Path:
    """Where ``get_deep_agent_sql_response`` writes ``<id>_attempt_XX.json`` (see ``deep_agent_runtime``)."""
    return (DEEP_AGENT_DIR / "generated_answers" / "deep_agent").resolve()


def _deep_agent_answer_json_exists(instance_id: str) -> bool:
    """True if any answer artifact exists for this benchmark id (resume / idempotent batch)."""
    stem = _safe_filename(instance_id)
    d = _deep_agent_generated_answers_dir()
    if not d.is_dir():
        return False
    return any(d.glob(f"{stem}_attempt_*.json"))


def _parse_args(argv: list[str]) -> tuple[str | None, str, int, str | None]:
    """Returns (prefix, mode, top_k, error_message). mode is sql-tool, deep-agent, or spider-agent."""
    args = list(argv)
    mode = "sql-tool"
    prefix = None
    top_k = 15
    if "--mode" in args:
        i = args.index("--mode")
        if i + 1 >= len(args):
            return None, mode, top_k, "--mode requires a value (sql-tool or deep-agent)"
        raw = args[i + 1].strip().lower().replace("_", "-")
        if raw not in ("sql-tool", "deep-agent", "spider-agent"):
            return None, mode, top_k, f"unknown --mode {raw!r}; use sql-tool, deep-agent, or spider-agent"
        mode = raw
    if "--top-k" in args:
        i = args.index("--top-k")
        if i + 1 >= len(args):
            return None, mode, top_k, "--top-k requires a positive integer value"
        try:
            top_k = int(args[i + 1])
        except ValueError:
            return None, mode, top_k, "--top-k must be an integer"
        if top_k <= 0:
            return None, mode, top_k, "--top-k must be > 0"
    if "--prefix" in args:
        i = args.index("--prefix")
        if i + 1 < len(args):
            prefix = args[i + 1]
    return prefix, mode, top_k, None


def run_single(
    question: str,
    top_k: int = 15,
    *,
    mode: str = "sql-tool",
    db: str | None = None,
    instance_id: str | None = None,
) -> str:
    """Generate SQL for one question (sql-tool or deep-agent).

    For deep-agent, ``db`` is the Spider2 database id (jsonl ``db`` field) used to pick
    the DuckDB schema; optional if only one schema exists in the file.

    Pass ``instance_id`` (e.g. jsonl ``instance_id``) so Deep Agent answer artifacts use
    the same filename stem as ``generated_sql/deep_agent/<instance_id>.sql``.
    """
    if mode == "deep-agent":
        from nemo_retriever.relational_db.benchmark.deep_agent.generate_sql import (
            get_deep_agent_sql_response,
        )

        payload = {"question": question, "user_id": None}
        if db is not None:
            payload["db"] = db
        if instance_id is not None:
            payload["instance_id"] = instance_id
        response = get_deep_agent_sql_response("debug", payload)
        return (response.get("sql_code") or "").strip() or ""

    elif mode == "sql-tool":
        from nemo_retriever.relational_db.benchmark.sql_tool.generate_sql import (
            get_sql_tool_response_top_k,
        )

        result = get_sql_tool_response_top_k(question, top_k=top_k)
        return (result.get("sql_code") or "").strip() or ""

    elif mode == "spider-agent":
        from nemo_retriever.relational_db.benchmark.spider_agent.generate_sql import (
            run_spider_agent_single,
        )

        sid = instance_id or os.environ.get("SPIDER_AGENT_INSTANCE_ID")
        if not sid:
            raise ValueError(
                "spider-agent single mode requires instance_id=... or env SPIDER_AGENT_INSTANCE_ID "
                "(e.g. local001)"
            )
        return (run_spider_agent_single(question, instance_id=sid, db=db) or "").strip()


def run_batch(
    jsonl_path: Path = SPIDER2_LITE_JSONL,
    output_dir: Path | None = None,
    top_k: int = 15,
    instance_id_prefix: str | None = None,
    *,
    mode: str = "sql-tool",
) -> None:
    """Loop over questions in jsonl; for each, write {instance_id}.sql into output_dir.

    When ``output_dir`` is omitted, **deep-agent** writes under
    ``benchmark/deep_agent/generated_sql/deep_agent/``; **sql-tool** writes under
    ``benchmark/sql_tool/generated_sql/sql_tool/``; **spider-agent** under
    ``benchmark/spider_agent/generated_sql/spider_agent/`` (upstream subprocess).

    **Resume:** skips work that already finished — **deep-agent** if
    ``generated_answers/deep_agent/<sanitized_id>_attempt_*.json`` exists (matches
    ``deep_agent_runtime``); **sql-tool** if ``<sanitized_id>.sql`` already exists in
    ``output_dir``. **spider-agent** runs one upstream job (resume is handled inside
    official ``run.py`` + harvest skip-existing).
    """
    if output_dir is None:
        output_dir = _canonical_batch_output_dir(mode)
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "spider-agent":
        from nemo_retriever.relational_db.benchmark.spider_agent.generate_sql import (
            run_spider_agent_benchmark,
        )

        print(
            "[spider-agent] Official Spider-Agent-Lite via SPIDER2_REPO_ROOT "
            "(Docker + credentials may be required; see methods/spider-agent-lite/README)."
        )
        print(f"  JSONL: {jsonl_path}")
        print(f"  Harvest .sql -> {output_dir}")
        if instance_id_prefix:
            print(f"  instance_id prefix: {instance_id_prefix!r}")
        try:
            run_spider_agent_benchmark(
                jsonl_path=jsonl_path,
                output_dir=output_dir,
                instance_id_prefix=instance_id_prefix,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    written = 0
    failed = 0
    skipped_prefix = 0
    skipped_existing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    label = "deep-agent" if mode == "deep-agent" else "sql-tool"
    if instance_id_prefix:
        print(
            f"[{label}] Processing instance_id prefix {instance_id_prefix!r} "
            f"from {jsonl_path} -> {output_dir}"
        )
    else:
        print(f"[{label}] Processing {total} questions from {jsonl_path} -> {output_dir}")

    print(f"Output directory (absolute): {output_dir}")

    for i, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [{i}/{total}] Skip invalid JSON: {e}")
            failed += 1
            continue

        instance_id = row.get("instance_id") or f"row_{i}"
        if instance_id_prefix and not instance_id.startswith(instance_id_prefix):
            skipped_prefix += 1
            continue

        question = row.get("question", "").strip()
        if not question:
            print(f"  [{i}/{total}] Skip {instance_id}: no question")
            failed += 1
            continue

        filename = _safe_filename(instance_id) + ".sql"
        out_path = output_dir / filename

        if mode == "deep-agent" and _deep_agent_answer_json_exists(instance_id):
            _ga = _deep_agent_generated_answers_dir()
            _stem = _safe_filename(instance_id)
            print(
                f"  [{i}/{total}] Skip {instance_id}: answer JSON exists "
                f"({_ga}/{_stem}_attempt_*.json)"
            )
            skipped_existing += 1
            continue

        if mode == "sql-tool" and out_path.is_file():
            print(
                f"  [{i}/{total}] Skip {instance_id}: {filename} already exists in {output_dir}"
            )
            skipped_existing += 1
            continue

        if mode == "deep-agent":
            db_id = row.get("db")
            print(f"  [{i}/{total}] {instance_id}")
            print(f"    question: {question}")
            print(f"    db (DuckDB schema): {db_id!r}")
            print("    ...", end=" ", flush=True)
        else:
            print(f"  [{i}/{total}] {instance_id} ...", end=" ", flush=True)
        try:
            if mode == "deep-agent":
                sql_code = run_single(
                    question,
                    top_k=top_k,
                    mode="deep-agent",
                    db=db_id,
                    instance_id=instance_id,
                )
            else:
                from nemo_retriever.relational_db.benchmark.sql_tool.generate_sql import (
                    get_sql_tool_response_top_k,
                )

                result = get_sql_tool_response_top_k(question, top_k=top_k)
                sql_code = (result.get("sql_code") or "").strip() or ""
            out_path.write_text(sql_code, encoding="utf-8")
            print(f" -> {filename}")
            written += 1
        except Exception as e:
            print(f" FAILED: {e}")
            failed += 1

    parts = [
        f"Written: {written}",
        f"failed: {failed}",
    ]
    if skipped_prefix:
        parts.append(f"skipped (prefix filter): {skipped_prefix}")
    if skipped_existing:
        parts.append(f"skipped (already had output): {skipped_existing}")
    print(f"Done. {', '.join(parts)}")


def main() -> None:
    if "--batch" in sys.argv:
        prefix, mode, top_k, err = _parse_args(sys.argv[1:])
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(2)
        run_batch(instance_id_prefix=prefix, mode=mode, top_k=top_k)
        return

    prefix, mode, top_k, err = _parse_args(sys.argv[1:])
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(2)

    # Single-question mode
    question = "how much stacks do we have"

    print("Mode:", mode)
    print("Question:", question)
    if mode == "sql-tool":
        print("Retrieving top_k =", top_k, "from LanceDB (nv-ingest-structured), then generating SQL...")
        print(
            "Batch output for sql-tool goes to:",
            _canonical_batch_output_dir("sql-tool"),
        )
    elif mode == "deep-agent":
        print("Using Deep Agent benchmark (DuckDB + agent; see benchmark/deep_agent/generate_sql.py)...")
        print(
            "Batch output for deep-agent goes to:",
            _canonical_batch_output_dir("deep-agent"),
        )
    else:
        print("Using upstream Spider-Agent-Lite (set SPIDER2_REPO_ROOT to a Spider2 clone).")
        print(
            "Batch output for spider-agent goes to:",
            _canonical_batch_output_dir("spider-agent"),
        )
    print()

    result = run_single(question, top_k=top_k, mode=mode, db=None)

    print("Result:")
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
