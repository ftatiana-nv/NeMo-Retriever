#!/usr/bin/env python3
"""Debug script: retrieve from LanceDB (structured table), then answer questions with SQL.

Modes (symmetrical layout under ``benchmark/``):

- **sql-tool** (default): LanceDB + RAG LLM → ``benchmark/sql_tool/generated_sql/sql_tool/``
- **deep-agent**: Deep Agent (``benchmark/deep_agent/generate_sql.py``) → ``benchmark/deep_agent/generated_sql/deep_agent/``

Single question (default, sql-tool):
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py

Deep-agent batch:
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch --mode deep-agent

Loop over benchmark/spider2-lite.jsonl (shared by sql-tool and deep-agent batch):
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch

Run only instances whose instance_id starts with a prefix (e.g. "local"):
  PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch --prefix local

Requires: LanceDB + LLM env for sql-tool; ``./spider2.duckdb`` (or ``DUCKDB_PATH``) + LLM for deep-agent batch.
"""

from __future__ import annotations

import json
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

# Shared Spider2-lite task list (all models / pipelines); not under sql_tool only.
SPIDER2_LITE_JSONL = _BENCHMARK / "spider2-lite.jsonl"

# Subfolder under ``generated_sql/`` so batch outputs are clearly separated by pipeline.
_OUTPUT_MODE_DIR = {"deep-agent": "deep_agent", "sql-tool": "sql_tool"}


def _canonical_batch_output_dir(mode: str) -> Path:
    """Default directory for batch ``.sql`` files: ``generated_sql/<sql_tool|deep_agent>/`` per mode."""
    sub = _OUTPUT_MODE_DIR.get(mode, "sql_tool")
    if mode == "deep-agent":
        return (DEEP_AGENT_DIR / "generated_sql" / sub).resolve()
    return (SQL_TOOL_DIR / "generated_sql" / sub).resolve()


def _safe_filename(instance_id: str) -> str:
    """Make instance_id safe for a filename (no path separators or reserved chars)."""
    return re.sub(r'[^\w\-.]', "_", instance_id) or "unknown"


def _parse_args(argv: list[str]) -> tuple[str | None, str, str | None]:
    """Returns (prefix, mode, error_message). mode is 'sql-tool' or 'deep-agent'."""
    args = list(argv)
    mode = "sql-tool"
    prefix = None
    if "--mode" in args:
        i = args.index("--mode")
        if i + 1 >= len(args):
            return None, mode, "--mode requires a value (sql-tool or deep-agent)"
        raw = args[i + 1].strip().lower().replace("_", "-")
        if raw not in ("sql-tool", "deep-agent"):
            return None, mode, f"unknown --mode {raw!r}; use sql-tool or deep-agent"
        mode = raw
    if "--prefix" in args:
        i = args.index("--prefix")
        if i + 1 < len(args):
            prefix = args[i + 1]
    return prefix, mode, None


def run_single(
    question: str,
    top_k: int = 15,
    *,
    mode: str = "sql-tool",
    db: str | None = None,
) -> str:
    """Generate SQL for one question (sql-tool or deep-agent).

    For deep-agent, ``db`` is the Spider2 database id (jsonl ``db`` field) used to pick
    the DuckDB schema; optional if only one schema exists in the file.
    """
    if mode == "deep-agent":
        from nemo_retriever.relational_db.benchmark.deep_agent.generate_sql import (
            get_deep_agent_sql_response,
        )

        payload = {"question": question, "user_id": None}
        if db is not None:
            payload["db"] = db
        r = get_deep_agent_sql_response("debug", payload)
        return (r.get("sql_code") or "").strip() or ""

    from nemo_retriever.relational_db.benchmark.deep_agent import generate_sql

    return generate_sql(question, top_k=top_k)


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
    ``benchmark/sql_tool/generated_sql/sql_tool/``.
    """
    if output_dir is None:
        output_dir = _canonical_batch_output_dir(mode)
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    skipped_prefix = 0

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
                    question, top_k=top_k, mode="deep-agent", db=db_id
                )
            else:
                from nemo_retriever.relational_db.benchmark.deep_agent import generate_sql

                sql_code = generate_sql(question, top_k=top_k)
            filename = _safe_filename(instance_id) + ".sql"
            out_path = output_dir / filename
            out_path.write_text(sql_code, encoding="utf-8")
            print(f" -> {filename}")
            written += 1
        except Exception as e:
            print(f" FAILED: {e}")
            failed += 1

    if skipped_prefix:
        print(f"Done. Written: {written}, failed: {failed}, skipped (prefix filter): {skipped_prefix}")
    else:
        print(f"Done. Written: {written}, failed: {failed}")


def main() -> None:
    if "--batch" in sys.argv:
        prefix, mode, err = _parse_args(sys.argv[1:])
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(2)
        run_batch(instance_id_prefix=prefix, mode=mode)
        return

    prefix, mode, err = _parse_args(sys.argv[1:])
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(2)

    # Single-question mode
    question = "how much stacks do we have"
    top_k = 15

    print("Mode:", mode)
    print("Question:", question)
    if mode == "sql-tool":
        print("Retrieving top_k =", top_k, "from LanceDB (nv-ingest-structured), then generating SQL...")
        print(
            "Batch output for sql-tool goes to:",
            _canonical_batch_output_dir("sql-tool"),
        )
    else:
        print("Using Deep Agent benchmark (DuckDB + agent; see benchmark/deep_agent/generate_sql.py)...")
        print(
            "Batch output for deep-agent goes to:",
            _canonical_batch_output_dir("deep-agent"),
        )
    print()

    result = run_single(question, top_k=top_k, mode=mode, db=None)

    print("Result:")
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
