"""Public entry point: `generate_sql` (SQL string) and `get_sql_tool_response_top_k` (full dict).

``generate_sql`` uses the OmniLite Deep Agent pipeline.
``get_sql_tool_response_top_k`` is the legacy LanceDB + single-LLM-call path, kept for
backwards compatibility.
"""

from nemo_retriever.tabular_data.retrieval.generate_sql import (
    get_sql_tool_response_top_k,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.main import get_agent_response


def generate_sql(query: str, top_k: int = 15, db_connector=None) -> str:  # noqa: ARG001
    """Generate SQL for a natural language query using the OmniLite Deep Agent.

    Args:
        query: Natural language question.
        top_k: Kept for backwards compatibility; ignored by the Deep Agent path.
        db_connector: Optional DB connector instance (e.g. ``DuckDB``).
            When provided it is forwarded to the ``execute_sql`` tool so
            results can be fetched without a separate connection setup.

    Returns:
        SQL string, or empty string if the agent could not construct one.
    """
    result = get_agent_response({"question": query, "db_connector": db_connector})
    return (result.get("sql_code") or "").strip()


__all__ = ["generate_sql", "get_sql_tool_response_top_k"]
