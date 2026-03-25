"""Benchmark harness for upstream Spider-Agent (Spider 2.0 ``methods/spider-agent-lite``).

Requires a clone of https://github.com/xlang-ai/Spider2 and ``SPIDER2_REPO_ROOT``.
Heavy dependencies (Docker, BigQuery/Snowflake creds for full runs) live upstream;
this package shells out to ``run.py`` and harvests ``.sql`` files into
``generated_sql/spider_agent/``.
"""

from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("run_spider_agent_benchmark", "run_spider_agent_single"):
        mod = importlib.import_module(
            "nemo_retriever.relational_db.benchmark.spider_agent.generate_sql"
        )
        return getattr(mod, name)
    if name == "harvest_sql_from_spider_output":
        from nemo_retriever.relational_db.benchmark.spider_agent.extract_sql import (
            harvest_sql_from_spider_output,
        )

        return harvest_sql_from_spider_output
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_spider_agent_benchmark",
    "run_spider_agent_single",
    "harvest_sql_from_spider_output",
]
