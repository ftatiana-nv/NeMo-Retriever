"""Deep Agent SQL generation (file layout mirrors ``benchmark/sql_tool/generate_sql.py``).

The **sql_tool** stack uses ``get_sql_tool_response_top_k`` (LanceDB + structured LLM).
This module is the **Deep Agent** path: ``get_deep_agent_sql_response`` (DuckDB +
Deep Agent + skills).

Both return a dict with ``sql_code``, ``answer``, and ``result`` when successful.
DuckDB agent construction and answer parsing live in ``deep_agent_runtime.py``.
"""

from __future__ import annotations

import os

from nemo_retriever.relational_db.benchmark.deep_agent.prompts import (
    format_deep_agent_user_prompt,
)


def get_deep_agent_sql_response(account_id: str, payload: dict) -> dict:
    """Run the Deep Agent on ``payload["question"]`` and return structured SQL output.

    This is the Deep Agent benchmark entry point (not the LanceDB ``sql_tool`` path).

    Parameters
    ----------
    account_id:
        Reserved for future account / tenancy routing (passed through for API symmetry).
    payload:
        Must include ``question``. Optional:

        - ``all_schemas`` (default: ``False`` when ``db``/``duckdb_schema`` is present, else ``True``): list/query **all** user schemas in the DuckDB
          file; ``db`` / ``duckdb_schema`` are not used to bind ``search_path``. Set
          ``all_schemas`` to ``False`` to pin one schema (use with ``db`` / ``duckdb_schema``
          or ``DEEP_AGENT_DUCKDB_SCHEMA``; see ``DEEP_AGENT_ALL_SCHEMAS`` env).
        - ``db`` / ``duckdb_schema``: Spider2 database id → DuckDB schema when **single-schema**
          mode is used.

        Optional: ``user_id`` for future group resolution.

    Returns
    -------
    dict
        ``{"sql_code": str, "answer": str, "result": ...}``
    """
    from nemo_retriever.relational_db.benchmark.deep_agent import deep_agent_runtime as rt

    question = payload.get("question")
    _user_id = payload.get("user_id")
    _ = account_id, _user_id  # reserved

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Default behavior: if a schema hint is provided by the benchmark row (``db``),
    # use single-schema tools unless caller explicitly overrides ``all_schemas``.
    duckdb_schema = payload.get("duckdb_schema") or payload.get("db")
    raw_all = payload.get("all_schemas", duckdb_schema is None)
    if isinstance(raw_all, bool):
        all_schemas = raw_all
    else:
        s = str(raw_all).strip().lower()
        all_schemas = s not in ("0", "false", "no", "off", "")

    agent = rt._get_sql_agent(
        base_dir,
        duckdb_schema=duckdb_schema,
        all_schemas=all_schemas,
    )

    prompt = format_deep_agent_user_prompt(question)

    print(
        "[deep-agent] question:\n"
        f"  {question}\n"
        "[deep-agent] schema:\n"
        f"  duckdb_schema/db={duckdb_schema!r}  all_schemas={all_schemas}"
    )

    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            parsed = rt._extract_structured_answer(result)
            if parsed is not None:
                result_dict = {
                    "sql_code": parsed.get("sql_code", ""),
                    "answer": parsed.get("answer", ""),
                    "result": parsed.get("result"),
                }
                rt._save_answer_json(base_dir, question, attempt, result_dict)
                return result_dict

            messages = result.get("messages") or []
            final_message = messages[-1] if messages else None
            raw_content = (
                getattr(final_message, "content", None)
                if final_message is not None
                else None
            )

            if raw_content is not None:
                result_dict = {
                    "sql_code": "",
                    "answer": raw_content,
                    "result": None,
                }
                rt._save_answer_json(base_dir, question, attempt, result_dict)
                return result_dict
        except Exception as e:  # noqa: PERF203
            print(
                f"Error in get_deep_agent_sql_response (attempt {attempt}/{max_retries}): {e}"
            )
            last_error = e

    return {
        "sql_code": "",
        "answer": f"Deep agent failed after {max_retries} attempts: {last_error}",
        "result": None,
    }

