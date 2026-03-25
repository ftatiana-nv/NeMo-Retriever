"""Harvest ``{instance_id}.sql`` from upstream Spider-Agent ``output/.../spider/result.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _extract_sql_from_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    block = re.search(r"```sql\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1).strip()
    t = text.strip()
    if t.upper().startswith("SELECT") or t.upper().startswith("WITH"):
        return t
    return ""


def sql_from_result_json(result_path: Path) -> str:
    """Best-effort SQL string from a Spider-Agent ``result.json``."""
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    for key in ("result", "final_answer", "answer"):
        v = data.get(key)
        if isinstance(v, str):
            s = _extract_sql_from_text(v)
            if s:
                return s

    # Trajectory / tool traces (structure varies by version)
    traj = data.get("trajectory")
    if isinstance(traj, list):
        for step in reversed(traj):
            if not isinstance(step, dict):
                continue
            if isinstance(step.get("content"), str):
                s = _extract_sql_from_text(step["content"])
                if s:
                    return s
    return ""


def _instance_id_from_spider_result_path(result_path: Path) -> str | None:
    """Infer ``instance_id`` from ``.../<instance_id>/spider/result.json``."""
    try:
        if result_path.parent.name != "spider":
            return None
        return result_path.parent.parent.name
    except Exception:
        return None


def harvest_sql_from_spider_output(
    spider_output_root: Path,
    dest_dir: Path,
    *,
    instance_id_prefix: str | None = None,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """Write ``dest_dir/<instance_id>.sql`` for each ``result.json`` found under ``spider_output_root``.

    Returns
    -------
    (written, skipped)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for result_path in sorted(spider_output_root.rglob("**/spider/result.json")):
        iid = _instance_id_from_spider_result_path(result_path)
        if not iid:
            continue
        if instance_id_prefix and not iid.startswith(instance_id_prefix):
            continue

        dst = dest_dir / f"{iid}.sql"
        if skip_existing and dst.is_file():
            skipped += 1
            continue

        sql = sql_from_result_json(result_path)
        dst.write_text(sql if sql else "-- (no SQL extracted from spider result.json)\n", encoding="utf-8")
        written += 1

    return written, skipped
