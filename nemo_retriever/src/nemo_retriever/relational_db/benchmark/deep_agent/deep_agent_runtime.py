"""Deep Agent construction and structured-answer parsing (DuckDB + LangChain SQL tools).

Used by ``generate_sql.get_deep_agent_sql_response``. Keeps heavy imports out of
the public ``generate_sql`` module surface unless this module is loaded.

Default DB path matches ``StructuredExtractParams(db_connection_string="./spider2.duckdb")``
and ``DUCKDB_PATH`` in ``.env.example``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from typing import Any, Iterable

# Do **not** call ``load_dotenv()`` here with no path. A cwd-only load can set empty
# ``LLM_API_KEY`` from a stray ``.env``; later ``generate_sql`` loads the repo ``.env``
# but python-dotenv does not override existing keys by default → unauthenticated
# requests and plain-text 404 on every model. Env is loaded in ``sql_tool/generate_sql``.
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text

from nemo_retriever.relational_db.benchmark.sql_tool.generate_sql import _make_llm

logging.basicConfig(level=logging.INFO)
logging.getLogger("deepagents").setLevel(logging.INFO)
logging.getLogger("langchain").setLevel(logging.INFO)

# One Deep Agent per resolved DuckDB schema, or one for all-schema mode (tool names stay
# ``sql_db_query``, etc. — no ``{schema}_sql_db_query`` prefixes).
_sql_agents: dict[str, Any] = {}

# System / catalog schemas to skip when building tools
_DUCKDB_SKIP_SCHEMAS = frozenset({"information_schema", "pg_catalog"})

# Internal delimiter for (schema, table) keys when listing all user schemas (unlikely in real names).
_SCHEMA_TABLE_JOINER = "."

# Injected SQL catalog for Deep Agent (see ``_deep_agent_sql_catalog_prompt``). Not read from ``os.environ``.
# ``names`` = comma-separated canonical keys (lightweight). ``full`` = full DDL via ``get_context()`` (large).
# ``off`` = do not inject.
DEEP_AGENT_SQL_CATALOG_MODE: str = "names"


def _canonical_multi_schema_table_key(raw: str, all_tables: set[str]) -> str | None:
    """Map tool input to internal key: accepts ``schema__/__table`` or ``schema.table``."""
    s = (raw or "").strip()
    if not s:
        return None
    if s in all_tables:
        return s
    # Strip one pair of outer quotes (models often pass "a"."b" or 'a'.'b' wrongly as one string)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        inner = s[1:-1].replace('""', '"')
        if inner in all_tables:
            return inner
        s = inner
    if s in all_tables:
        return s
    if _SCHEMA_TABLE_JOINER in s:
        return s if s in all_tables else None
    # Natural ``schema.table`` (first dot splits schema from table; same as many LLM outputs)
    if "." in s:
        sch, _, tbl = s.partition(".")
        sch, tbl = sch.strip(), tbl.strip()
        if sch and tbl:
            cand = _make_schema_table_key(sch, tbl)
            if cand in all_tables:
                return cand
    return None


def _duckdb_quote_ident(ident: str) -> str:
    """Double-quote a DuckDB identifier (escape embedded quotes)."""
    return '"' + ident.replace('"', '""') + '"'


def _make_schema_table_key(schema: str, table: str) -> str:
    return f"{schema}{_SCHEMA_TABLE_JOINER}{table}"


def _split_schema_table_key(key: str) -> tuple[str, str]:
    if _SCHEMA_TABLE_JOINER not in key:
        raise ValueError(
            f"Invalid multi-schema table key {key!r} (expected 'schema{_SCHEMA_TABLE_JOINER}table')"
        )
    return key.split(_SCHEMA_TABLE_JOINER, 1)


def _deep_agent_all_schemas_enabled() -> bool:
    """When True (default), expose all user schemas so tools search the whole DuckDB file."""
    v = os.environ.get("DEEP_AGENT_ALL_SCHEMAS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _deep_agent_skill_dirs(base_dir: str) -> list[str] | None:
    """Skill folders loaded into the agent (``schema-exploration``, ``query-writing``).

    **Default on.** Set ``DEEP_AGENT_LOAD_SKILLS=0`` to disable and save prompt tokens
    if the model context is tight (e.g. 8k total window).
    """
    v = os.environ.get("DEEP_AGENT_LOAD_SKILLS", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return None
    return [
        os.path.join(base_dir, "skills", "schema-exploration"),
        os.path.join(base_dir, "skills", "query-writing"),
    ]


class DuckDBLangChainSQLDatabase(SQLDatabase):
    """``SQLDatabase`` that does not use SQLAlchemy ``MetaData.reflect()``.

    ``duckdb-engine`` subclasses the PostgreSQL dialect; reflection runs
    ``pg_catalog`` queries that DuckDB does not fully implement (e.g. missing
    ``pg_collation``), so LangChain's default path yields **zero** tools after
    every schema fails. Table metadata is built from DuckDB's ``duckdb_columns()``
    instead. This is independent of Deep Agent **skills** (filesystem prompts);
    those do not register SQL tools.
    """

    def __init__(
        self,
        engine,
        *,
        schema: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("lazy_table_reflection", True)
        kwargs.setdefault("sample_rows_in_table_info", 0)
        kwargs.setdefault("indexes_in_table_info", False)
        kwargs.setdefault("view_support", False)
        super().__init__(engine, schema=schema, **kwargs)

    def get_table_info(
        self,
        table_names: list[str] | None = None,
        get_col_comments: bool = False,
    ) -> str:
        """Build CREATE TABLE-style text from ``duckdb_columns()`` (no SQLAlchemy reflect)."""
        all_table_names = list(self.get_usable_table_names())
        if table_names is not None:
            missing = set(table_names).difference(all_table_names)
            if missing:
                raise ValueError(f"table_names {missing} not found in database")
            all_table_names = list(table_names)

        sch = self._schema
        blocks: list[str] = []
        with self._engine.connect() as conn:
            for tname in sorted(all_table_names):
                if self._custom_table_info and tname in self._custom_table_info:
                    blocks.append(self._custom_table_info[tname])
                    continue

                rows = conn.execute(
                    text(
                        """
                        SELECT column_name, data_type
                        FROM duckdb_columns()
                        WHERE schema_name = :sch AND table_name = :tn
                        ORDER BY column_index NULLS LAST, column_name
                        """
                    ),
                    {"sch": sch, "tn": tname},
                ).fetchall()

                if not rows:
                    rows = conn.execute(
                        text(
                            """
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_schema = :sch AND table_name = :tn
                            ORDER BY ordinal_position
                            """
                        ),
                        {"sch": sch, "tn": tname},
                    ).fetchall()

                if not rows:
                    blocks.append(f'/* no columns found for "{tname}" */')
                    continue

                col_lines = ",\n  ".join(f'"{r[0]}" {r[1]}' for r in rows)
                blocks.append(f'CREATE TABLE "{tname}" (\n  {col_lines}\n);')

        return "\n\n".join(blocks)


class DuckDBAllSchemasSQLDatabase(DuckDBLangChainSQLDatabase):
    """``SQLDatabase`` over **all** user schemas: list/schema tools see every table.

    Sets ``SQLDatabase._schema`` to ``None`` so DuckDB does **not** run
    ``SET search_path TO ...`` (LangChain's default for a single schema). The model
    must use **schema-qualified** identifiers in SQL, e.g.
    ``"some_schema"."customers"``, matching the ``CREATE TABLE`` snippets here.

    Table keys passed to ``sql_db_schema`` / ``get_table_info`` look like
    ``schema_name__/__table_name`` (see ``_SCHEMA_TABLE_JOINER``); use the qualified
    ``CREATE TABLE`` form from ``sql_db_schema`` in real SQL.
    """

    def __init__(
        self,
        engine,
        *,
        user_schemas: list[str],
        **kwargs: Any,
    ) -> None:
        if not user_schemas:
            raise ValueError("user_schemas must be non-empty for DuckDBAllSchemasSQLDatabase")
        kwargs.setdefault("lazy_table_reflection", True)
        kwargs.setdefault("sample_rows_in_table_info", 0)
        kwargs.setdefault("indexes_in_table_info", False)
        kwargs.setdefault("view_support", False)
        # Bootstrap SQLDatabase with a real schema so inspector-based init succeeds.
        first = user_schemas[0]
        super().__init__(engine, schema=first, **kwargs)
        # Critical: do not pin search_path — require qualified names in queries.
        self._schema = None
        self._duckdb_user_schemas = tuple(user_schemas)

        insp = inspect(engine)
        keys: list[str] = []
        for sch in user_schemas:
            try:
                tables = insp.get_table_names(schema=sch) or []
            except Exception as e:  # noqa: PERF203
                print(f"Warning: could not list tables for schema {sch!r}: {e}")
                tables = []
            for t in tables:
                keys.append(_make_schema_table_key(sch, t))

        self._all_tables = set(keys)
        u = list(self.get_usable_table_names())
        self._usable_tables = set(u) if u else self._all_tables

    def get_usable_table_names(self) -> Iterable[str]:
        """Return every (schema, table) key — must match ``sql_db_list_tables`` output."""
        return sorted(self._all_tables)

    def get_table_info(
        self,
        table_names: list[str] | None = None,
        get_col_comments: bool = False,
    ) -> str:
        """Like :meth:`DuckDBLangChainSQLDatabase.get_table_info` but per (schema, table) key."""
        all_table_names = list(self.get_usable_table_names())
        if table_names is not None:
            resolved: list[str] = []
            missing_raw: list[str] = []
            for raw in table_names:
                canon = _canonical_multi_schema_table_key(raw, self._all_tables)
                if canon is None:
                    missing_raw.append(raw)
                else:
                    resolved.append(canon)
            if missing_raw:
                raise ValueError(
                    "table_names not found in database (use keys from sql_db_list_tables, or "
                    f'"schema.table" with the real schema and table names): {missing_raw!r}'
                )
            # Dedupe while preserving order
            seen: set[str] = set()
            all_table_names = []
            for k in resolved:
                if k not in seen:
                    seen.add(k)
                    all_table_names.append(k)

        blocks: list[str] = []
        with self._engine.connect() as conn:
            for tname in sorted(all_table_names):
                if self._custom_table_info and tname in self._custom_table_info:
                    blocks.append(self._custom_table_info[tname])
                    continue

                sch, tn = _split_schema_table_key(tname)

                rows = conn.execute(
                    text(
                        """
                        SELECT column_name, data_type
                        FROM duckdb_columns()
                        WHERE schema_name = :sch AND table_name = :tn
                        ORDER BY column_index NULLS LAST, column_name
                        """
                    ),
                    {"sch": sch, "tn": tn},
                ).fetchall()

                if not rows:
                    rows = conn.execute(
                        text(
                            """
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_schema = :sch AND table_name = :tn
                            ORDER BY ordinal_position
                            """
                        ),
                        {"sch": sch, "tn": tn},
                    ).fetchall()

                if not rows:
                    blocks.append(f'/* no columns found for {tname!r} */')
                    continue

                col_lines = ",\n  ".join(f'"{r[0]}" {r[1]}' for r in rows)
                qual = f"{_duckdb_quote_ident(sch)}.{_duckdb_quote_ident(tn)}"
                blocks.append(f"CREATE TABLE {qual} (\n  {col_lines}\n);")

        return "\n\n".join(blocks)


class DuckDBSQLDatabaseToolkit(SQLDatabaseToolkit):
    """`SQLDatabaseToolkit` with tool descriptions that match DuckDB multi-schema keys.

    Aligns with LangChain's pattern of passing `get_context()` into the agent; we also
    tighten descriptions so the model does not treat `sql_db_list_tables` output as bare
    table names when every key is ``schema{SCHEMA_JOINER}table``.
    """

    def get_tools(self) -> list[Any]:
        tools = super().get_tools()
        if not isinstance(self.db, DuckDBAllSchemasSQLDatabase):
            return tools
        delim = _SCHEMA_TABLE_JOINER
        for t in tools:
            name = getattr(t, "name", None)
            if name == "sql_db_list_tables":
                t.description = (
                    "Input must be an empty string. Returns a comma-separated list of "
                    "ALL table keys in this DuckDB file (every user schema). "
                    f"Each key uses SCHEMA{delim}TABLE (one delimiter between schema and table). "
                    "You may pass either that form or schema.table to sql_db_schema. In SQL, use "
                    '"SCHEMA"."TABLE". Do not invent table names unless they appear in this list.'
                )
            elif name == "sql_db_schema":
                t.description = (
                    "Input is a comma-separated list of tables to describe. Use keys from "
                    f"sql_db_list_tables (schema{delim}table) OR the equivalent schema.table form "
                    "(e.g. complex_oracle.sales). Output is CREATE TABLE DDL with column types."
                )
            elif name == "sql_db_query":
                t.description = (
                    "Execute a DuckDB SELECT. Reference only tables/columns from sql_db_schema "
                    "for keys you requested. Use schema-qualified identifiers as in the DDL."
                )
        return tools


def _deep_agent_sql_catalog_prompt(db: SQLDatabase) -> str | None:
    """Extra system prompt: real table names / optional full DDL (``SQLDatabase.get_context``)."""
    mode = (DEEP_AGENT_SQL_CATALOG_MODE or "names").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return None
    try:
        ctx = db.get_context()
    except Exception as e:
        print(f"Warning: could not build SQL catalog prompt: {e}")
        return None
    if mode == "full":
        return (
            "## DuckDB catalog (ground truth — never invent tables or columns)\n\n"
            f"{ctx.get('table_info', '')}\n"
        )
    return (
        "## DuckDB catalog (ground truth — never invent table or column names)\n\n"
        "- Call `sql_db_list_tables` then `sql_db_schema` before writing SQL.\n"
        f"- Tool keys look like SCHEMA{_SCHEMA_TABLE_JOINER}TABLE; you may also pass schema.table "
        'to `sql_db_schema`. In SQL use `"schema"."table"`.\n\n'
        f"**All table keys in this database:** {ctx.get('table_names', '')}\n"
    )


def _duckdb_file_path() -> Path:
    """Resolve DuckDB file path (default ``./spider2.duckdb``, same as structured ingest)."""
    raw = os.environ.get("DUCKDB_PATH", "./spider2.duckdb")
    return Path(raw).expanduser().resolve()


def _duckdb_sqlalchemy_url(path: Path) -> str:
    """SQLAlchemy URL for ``duckdb-engine`` (absolute path)."""
    return f"duckdb:///{path.as_posix()}"


def _resolve_duckdb_schema(requested: str | None, user_schemas: list[str]) -> str:
    """Map payload ``db`` / env to a real DuckDB schema name (Spider2: one schema per database)."""
    if not user_schemas:
        raise ValueError("No user schemas found in DuckDB (empty database?)")

    env_schema = os.environ.get("DEEP_AGENT_DUCKDB_SCHEMA", "").strip()
    hint = (requested or env_schema or "").strip() or None

    if hint is None:
        if len(user_schemas) == 1:
            chosen = user_schemas[0]
            print(
                f"Deep Agent: using sole DuckDB schema {chosen!r} "
                "(set DEEP_AGENT_DUCKDB_SCHEMA or pass payload['db'] if wrong)."
            )
            return chosen
        raise ValueError(
            "Multiple DuckDB schemas exist; pick one for SQL tools. "
            "Set env DEEP_AGENT_DUCKDB_SCHEMA, or pass payload['duckdb_schema'] / payload['db'] "
            "(e.g. spider2-lite jsonl \"db\" field). "
            f"Available: {user_schemas[:20]}{'...' if len(user_schemas) > 20 else ''}"
        )

    if hint in user_schemas:
        return hint

    hint_lower = hint.lower()
    for s in user_schemas:
        if s.lower() == hint_lower:
            return s

    # Match canonical schema aliases (e.g. Spider2 schemas like spider2."E_commerce"
    # should resolve from hint "E_commerce") without broad substring matching.
    def _schema_aliases(s: str) -> list[str]:
        out = [s]
        t = s.strip()
        out.append(t)
        out.append(t.lower())
        if len(t) >= 2 and t[0] == t[-1] == '"':
            inner = t[1:-1]
            out.extend([inner, inner.lower()])
        if "." in t:
            tail = t.rsplit(".", 1)[-1].strip()
            out.extend([tail, tail.lower()])
            if len(tail) >= 2 and tail[0] == tail[-1] == '"':
                inner_tail = tail[1:-1]
                out.extend([inner_tail, inner_tail.lower()])
        # Keep insertion order; drop empties.
        uniq: list[str] = []
        seen: set[str] = set()
        for x in out:
            if x and x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    exact_hits: list[str] = []
    for s in user_schemas:
        aliases = _schema_aliases(s)
        if hint in aliases or hint_lower in aliases:
            exact_hits.append(s)
    if len(exact_hits) == 1:
        return exact_hits[0]
    if len(exact_hits) > 1:
        raise ValueError(
            f"Ambiguous schema hint {hint!r}; matches {exact_hits[:10]}. "
            "Pass an exact schema name from the list."
        )

    raise ValueError(
        f"Could not resolve DuckDB schema from {hint!r}. "
        f"Try an exact name from: {user_schemas[:30]}"
    )


def _get_sql_agent(
    base_dir: str,
    duckdb_schema: str | None = None,
    *,
    all_schemas: bool = True,
):
    """
    Create and cache the Deep Agent with a DuckDB SQL toolkit.

    - **All user schemas** (default when ``all_schemas=True`` and
      ``DEEP_AGENT_ALL_SCHEMAS`` is not disabled): tools list every table in every
      user schema; **no** ``search_path`` — SQL must use qualified
      ``"schema"."table"`` names (see ``DuckDBAllSchemasSQLDatabase``). ``db`` /
      ``duckdb_schema`` in the payload are ignored for binding (use for logging only).
    - **Single schema** (``all_schemas=False``, or ``DEEP_AGENT_ALL_SCHEMAS=0``):
      bind one schema via ``db`` / ``duckdb_schema`` / env; ``search_path`` is set.

    LangChain tools keep canonical names ``sql_db_query``, ``sql_db_list_tables``,
    ``sql_db_schema``, ``sql_db_query_checker`` — matching ``AGENTS.md`` and ``skills/``.
    A `DuckDBSQLDatabaseToolkit` tightens tool descriptions for multi-schema keys; see
    ``DEEP_AGENT_SQL_CATALOG_MODE`` in this module for injecting ``get_context()`` into the system prompt
    (LangChain / `deepagents` pattern: ground the model in real tables —
    https://github.com/langchain-ai/deepagents ).
    Caching is per resolved schema (or the all-schemas sentinel), ``all_schemas``, and catalog mode.
    """
    global _sql_agents

    try:
        import duckdb_engine  # noqa: F401  # registers SQLAlchemy dialect
    except ImportError as e:
        raise ImportError(
            "duckdb-engine is required for Deep Agent + DuckDB. "
            "Install: pip install 'duckdb-engine>=0.13.0'"
        ) from e

    db_path = _duckdb_file_path()
    if not db_path.is_file():
        raise FileNotFoundError(
            f"DuckDB database not found: {db_path}. "
            "Set DUCKDB_PATH or create the file (see relational_db/connectors/SPIDER2_SETUP.md)."
        )

    duckdb_uri = _duckdb_sqlalchemy_url(db_path)

    engine = create_engine(duckdb_uri)
    inspector = inspect(engine)

    available_schemas = inspector.get_schema_names()
    user_schemas = [s for s in available_schemas if s.lower() not in _DUCKDB_SKIP_SCHEMAS]

    all_table_count = 0
    for schema in user_schemas:
        try:
            tables = inspector.get_table_names(schema=schema)
            all_table_count += len(tables or [])
        except Exception as e:
            print(f"Warning: Could not access schema {schema}: {e}")

    print(
        f"DuckDB {db_path}: {all_table_count} tables across {len(user_schemas)} schema(s)"
    )

    use_all_user_schemas = (
        bool(all_schemas)
        and bool(available_schemas)
        and _deep_agent_all_schemas_enabled()
        and len(user_schemas) >= 1
    )

    if use_all_user_schemas:
        resolved = "__ALL_USER_SCHEMAS__"
        print(
            f"Deep Agent: all user schemas mode ({len(user_schemas)} schemas, "
            f"{all_table_count} tables). "
            "Qualify table names in SQL as shown in sql_db_schema (set "
            "DEEP_AGENT_ALL_SCHEMAS=0 to bind one schema via db/DEEP_AGENT_DUCKDB_SCHEMA)."
        )
    else:
        resolved = _resolve_duckdb_schema(duckdb_schema, user_schemas)

    skill_dirs = _deep_agent_skill_dirs(base_dir)
    _cat_mode = (DEEP_AGENT_SQL_CATALOG_MODE or "names").strip().lower()
    cache_key = f"{resolved}::skills={bool(skill_dirs)}::all={use_all_user_schemas}::cat={_cat_mode}"
    if cache_key in _sql_agents:
        return _sql_agents[cache_key]

    llm = _make_llm()
    shared_engine = create_engine(duckdb_uri, pool_pre_ping=True)

    if use_all_user_schemas:
        db_for_schema = DuckDBAllSchemasSQLDatabase(
            shared_engine,
            user_schemas=user_schemas,
        )
    else:
        db_for_schema = DuckDBLangChainSQLDatabase(
            shared_engine,
            schema=resolved,
        )
    toolkit = DuckDBSQLDatabaseToolkit(db=db_for_schema, llm=llm)
    all_sql_tools = toolkit.get_tools()

    print(
        f"Deep Agent SQL tools for {resolved!r}: "
        f"{[getattr(t, 'name', '?') for t in all_sql_tools]}"
    )

    sql_catalog_prompt = _deep_agent_sql_catalog_prompt(db_for_schema)
    if sql_catalog_prompt:
        print(
            f"Deep Agent: SQL catalog system prompt = {_cat_mode!r} "
            f"({len(sql_catalog_prompt)} chars; set DEEP_AGENT_SQL_CATALOG_MODE='off' in deep_agent_runtime to disable)."
        )

    print(
        f"Deep Agent: skills loaded = {bool(skill_dirs)} "
        "(set DEEP_AGENT_LOAD_SKILLS=0 to disable; use a long-context LLM_MODEL + DEEP_AGENT_MAX_TOKENS if prompts are large)."
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=sql_catalog_prompt,
        memory=[os.path.join(base_dir, "AGENTS.md")],
        skills=skill_dirs,
        tools=all_sql_tools,
        subagents=[],
        backend=FilesystemBackend(root_dir=base_dir),
    )

    _sql_agents[cache_key] = agent
    return agent


def _answer_artifact_path(base_dir: str, question: str, attempt: int) -> str:
    """Return per-question answer artifact path under ``generated_answers/deep_agent``."""
    # Keep a short, stable filename prefix from the beginning of the question.
    q = (question or "").strip().lower()
    prefix = re.sub(r"[^a-z0-9]+", "_", q).strip("_")
    prefix = (prefix[:24] or "question")
    filename = f"{prefix}_attempt_{attempt:02d}.json"
    return os.path.join(base_dir, "generated_answers", "deep_agent", filename)


def _save_answer_json(base_dir: str, question: str, attempt: int, answer: dict) -> None:
    """
    Persist the structured SQL answer to ``generated_answers/deep_agent/<question>_attempt_XX.json``
    for easier inspection/debugging.
    """
    try:
        answer_path = _answer_artifact_path(base_dir, question, attempt)
        os.makedirs(os.path.dirname(answer_path), exist_ok=True)
        with open(answer_path, "w", encoding="utf-8") as f:
            json.dump(answer, f, ensure_ascii=False)
    except Exception as e:  # noqa: PERF203
        print(f"Warning: Could not save answer.json: {e}")


_REQUIRED_ANSWER_KEYS = frozenset({"sql_code", "answer", "result"})


def _extract_json_answer_object(content: str) -> dict | None:
    """Parse a ``{sql_code, answer, result}`` object from a message.

    Models often prefix JSON with prose (e.g. "The final answer is:\\n\\n{...}").
    ``json.loads`` on the whole string fails; ``JSONDecoder.raw_decode`` finds the
    first valid embedded object.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(obj.keys()):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, _end = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(
                    obj.keys()
                ):
                    return obj
            except json.JSONDecodeError:
                pass
        i += 1
    return None


def _extract_sql_from_tool_shaped_json(content: str) -> dict | None:
    """Recover SQL when the model prints ``{"name":"sql_db_query","parameters":{...}}`` in
    ``content`` instead of using real ``tool_calls`` (often paired with ``finish_reason=length``).

    Some models stream a *text* representation of a tool call; the JSON may be **invalid**
    if truncated mid-string — then nothing can be recovered.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, _end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if not isinstance(obj, dict):
            i += 1
            continue
        name = obj.get("name")
        params = obj.get("parameters")
        if name not in ("sql_db_query", "sql_db_query_checker") or not isinstance(
            params, dict
        ):
            i += 1
            continue
        q = params.get("query")
        if not isinstance(q, str) or not q.strip():
            i += 1
            continue
        return {
            "sql_code": q.strip(),
            "answer": (
                "Recovered from assistant message text (expected structured tool_calls; "
                "model emitted JSON instead). If SQL is garbage or cut off, check "
                "finish_reason=length and increase max output tokens."
            ),
            "result": None,
        }
    return None


def _extract_structured_answer(result: dict) -> dict | None:
    """
    Scan all Deep Agent messages from the end and find the first one that
    contains a JSON object with sql_code, answer, and result.

    If no such JSON object is found, apply a best-effort heuristic parser
    to markdown-style answers that include:
    - a ```sql ... ``` code block, and
    - a human-readable answer and/or result section.
    """
    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        obj = None
        if isinstance(content, str):
            obj = _extract_json_answer_object(content)
        elif isinstance(content, dict):
            obj = content if _REQUIRED_ANSWER_KEYS.issubset(content.keys()) else None

        if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(obj.keys()):
            return obj

        if isinstance(content, str):
            toolish = _extract_sql_from_tool_shaped_json(content)
            if toolish is not None:
                return toolish

        # If not JSON, try to heuristically parse markdown-style content
        if isinstance(content, str):
            heuristic = _parse_markdown_answer(content)
            if heuristic is not None:
                return heuristic

    return None


def _parse_markdown_answer(text: str) -> dict | None:
    """
    Best-effort parser for markdown-style answers of the form:

    ### Final Answer

    **SQL Code Executed:**
    ```sql
    SELECT ...
    ```

    **Result:**
    - ...

    **Answer:**
    some explanation...
    """
    sql_code = None
    answer = None
    result_value: object | None = None

    # 1) Extract SQL between ```sql and next ```
    start = text.find("```sql")
    if start != -1:
        start = text.find("\n", start)
        if start != -1:
            end = text.find("```", start)
            if end != -1:
                sql_code_block = text[start:end]
                sql_code = sql_code_block.strip()

    # 2) Extract answer text after "**Answer:**" (if present)
    answer_marker = "**Answer:**"
    idx = text.find(answer_marker)
    if idx != -1:
        answer = text[idx + len(answer_marker) :].strip()

    # 3) Extract a numeric result from the "Result" section if present
    #    We look between "**Result:**" and "**Answer:**" (or end of text)
    result_marker = "**Result:**"
    r_idx = text.find(result_marker)
    if r_idx != -1:
        r_start = r_idx + len(result_marker)
        r_end = text.find("**Answer:**", r_start)
        if r_end == -1:
            r_end = len(text)
        result_section = text[r_start:r_end]
        # Try to find a number in the result section
        m = re.search(r"-?\d+(\.\d+)?", result_section)
        if m:
            num_str = m.group(0)
            try:
                result_value = float(num_str)
            except ValueError:
                result_value = num_str
        else:
            # Fallback: keep the raw section as result if non-empty
            if result_section.strip():
                result_value = result_section.strip()

    # If we at least have SQL and some answer text, return a dict
    if sql_code and answer:
        return {
            "sql_code": sql_code,
            "answer": answer,
            "result": result_value,
        }

    return None
