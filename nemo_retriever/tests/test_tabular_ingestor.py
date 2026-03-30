"""Unit tests for the tabular ingestion flow in BatchIngestor.

All external databases (DuckDB, Neo4j, etc.) are replaced with lightweight
in-process stubs so the tests run without any infrastructure.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("ray")

from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.params import TabularExtractParams
from nemo_retriever.tabular_data.ingestion.extract_data import (
    data_for_populate_tabular,
    store_relational_db_in_neo4j,
)


# ── Ray / Ray-Data stubs ───────────────────────────────────────────────────────


class _DummyClusterResources:
    def total_cpu_count(self) -> int:
        return 4

    def total_gpu_count(self) -> int:
        return 0

    def available_cpu_count(self) -> int:
        return 4

    def available_gpu_count(self) -> int:
        return 0


@pytest.fixture()
def batch_ingestor(monkeypatch):
    """Return a BatchIngestor with all Ray / Ray-Data side-effects patched out."""
    dummy_ctx = SimpleNamespace(enable_rich_progress_bars=False, use_ray_tqdm=True)

    monkeypatch.setattr("nemo_retriever.ingest_modes.batch.ray.init", lambda **kwargs: None)
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.rd.DataContext.get_current",
        lambda: dummy_ctx,
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.gather_cluster_resources",
        lambda _ray: _DummyClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_requested_plan",
        lambda cluster_resources, allow_no_gpu=False: {"plan": "dummy"},
    )
    return BatchIngestor(documents=[])


# ── Fake DB row data matching the column names DuckDB connector returns ────────

_FAKE_TABLES = pd.DataFrame(
    {
        "database": ["mydb"],
        "schema": ["public"],
        "table_name": ["orders"],
    }
)

_FAKE_COLUMNS = pd.DataFrame(
    {
        "database": ["mydb"],
        "schema": ["public"],
        "table_name": ["orders"],
        "column_name": ["id"],
        "ordinal_position": [1],
        "data_type": ["INTEGER"],
        "is_nullable": ["NO"],
    }
)

_FAKE_VIEWS = pd.DataFrame(
    {
        "database": ["mydb"],
        "schema": ["public"],
        "table_name": ["v_orders"],
        "view_definition": ["SELECT * FROM orders"],
    }
)

# DuckDB connector returns these column names for PKs (database / schema, not *_name variants)
_FAKE_PKS = pd.DataFrame(
    {
        "database": ["mydb"],
        "schema": ["public"],
        "table_name": ["orders"],
        "column_name": ["id"],
        "ordinal_position": [1],
    }
)

# No FKs in this schema — empty DataFrame matching connector column set
_FAKE_FKS = pd.DataFrame(
    columns=[
        "database",
        "schema",
        "table_name",
        "column_name",
        "referenced_schema",
        "referenced_table",
        "referenced_column",
    ]
)


class _DummyDuckDB:
    """Drop-in replacement for DuckDB that returns pre-canned DataFrames."""

    def __init__(self, connection_string: str) -> None:
        pass

    def get_tables(self) -> pd.DataFrame:
        return _FAKE_TABLES.copy()

    def get_columns(self) -> pd.DataFrame:
        return _FAKE_COLUMNS.copy()

    def get_views(self) -> pd.DataFrame:
        return _FAKE_VIEWS.copy()

    def get_pks(self) -> pd.DataFrame:
        return _FAKE_PKS.copy()

    def get_fks(self) -> pd.DataFrame:
        return _FAKE_FKS.copy()

    def get_queries(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["end_time", "query_text"])


# ── Tests ──────────────────────────────────────────────────────────────────────

EXPECTED_DATA_KEYS = {"tables", "columns", "views", "pks", "fks"}


def test_pull_tabular_db_entities(batch_ingestor, monkeypatch):
    """pull_tabular_db_entities returns the expected schema dict, content, and tolerates params=None."""
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.ingestion.extract_data.DuckDB",
        _DummyDuckDB,
    )

    # ── with explicit params ───────────────────────────────────────────────────
    data = batch_ingestor.pull_tabular_db_entities(params=TabularExtractParams(connection_string="dummy.duckdb"))

    assert isinstance(data, dict)
    assert set(data.keys()) == EXPECTED_DATA_KEYS
    for key in EXPECTED_DATA_KEYS:
        assert isinstance(data[key], pd.DataFrame), f"data['{key}'] must be a DataFrame"

    assert data["tables"]["table_name"].iloc[0] == "orders"
    assert data["columns"]["column_name"].iloc[0] == "id"
    assert data["views"]["table_name"].iloc[0] == "v_orders"
    assert data["fks"].empty

    # ── params=None falls back to defaults and still returns the correct shape ─
    data_default = batch_ingestor.pull_tabular_db_entities(params=None)
    assert set(data_default.keys()) == EXPECTED_DATA_KEYS
    for key in EXPECTED_DATA_KEYS:
        assert isinstance(data_default[key], pd.DataFrame)


# ── data_for_populate_tabular ──────────────────────────────────────────────────


def test_data_for_populate_tabular(monkeypatch):
    """data_for_populate_tabular returns all required keys as DataFrames and applies normalization."""
    raw_tables = pd.DataFrame(
        {
            "database": ["mydb"],
            "schema": ["public"],
            "table_name": ["orders"],
            "owner": ["dba"],  # normalize_tables should drop this
        }
    )
    raw_columns = pd.DataFrame(
        {
            "database": ["mydb"],
            "schema": ["public"],
            "table_name": ["orders"],
            "column_name": ["id"],
            "ordinal_position": ["1"],  # string; normalize_columns should coerce to Int16
            "data_type": ["INTEGER"],
            "is_nullable": ["NO"],
        }
    )
    raw_views = pd.DataFrame(
        {"database": ["mydb"], "schema": ["public"], "table_name": ["v_orders"], "view_definition": ["SELECT 1"]}
    )
    raw_pks = pd.DataFrame(columns=["database", "schema", "table_name", "column_name"])
    raw_fks = pd.DataFrame(columns=["database", "schema", "table_name", "column_name"])
    raw_queries = pd.DataFrame(columns=["end_time", "query_text"])

    monkeypatch.setattr(
        "nemo_retriever.tabular_data.ingestion.extract_data.create_dataframe",
        lambda settings: (raw_tables, raw_columns, raw_views, raw_queries, raw_pks, raw_fks),
    )

    data = data_for_populate_tabular({"connection_string": "dummy.duckdb"})

    # required keys, all DataFrames
    assert set(data.keys()) == EXPECTED_DATA_KEYS
    for key in EXPECTED_DATA_KEYS:
        assert isinstance(data[key], pd.DataFrame)

    # normalize_tables: owner dropped, dtypes applied
    assert "owner" not in data["tables"].columns
    assert str(data["tables"]["database"].dtype) == "category"
    assert str(data["tables"]["table_name"].dtype) == "string"

    # normalize_columns: string "1" coerced to Int16
    assert str(data["columns"]["ordinal_position"].dtype) == "Int16"
    assert data["columns"]["ordinal_position"].iloc[0] == 1

    # views is a raw pass-through; pks/fks are empty; queries not surfaced
    assert len(data["views"]) == 1
    assert data["pks"].empty
    assert data["fks"].empty
    assert "queries" not in data


# ── store_relational_db_in_neo4j ───────────────────────────────────────────────


def test_store_relational_db_in_neo4j_delegates_to_populate(monkeypatch):
    """store_relational_db_in_neo4j calls populate_tabular_data with the exact args."""
    calls: list[dict] = []

    # Patch get_neo4j_conn *before* write_to_graph is imported so that the
    # module-level `conn = get_neo4j_conn()` calls in schemas_dal and db_dal
    # do not require NEO4J_URI to be set in the environment.
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.neo4j.get_neo4j_conn",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.ingestion.write_to_graph.populate_tabular_data",
        lambda data, num_workers, dialect: calls.append({"data": data, "num_workers": num_workers, "dialect": dialect}),
    )

    dummy_data = {k: pd.DataFrame() for k in ("tables", "columns", "views", "pks", "fks")}
    store_relational_db_in_neo4j(data=dummy_data, neo4j_conn=None)

    assert len(calls) == 1
    assert calls[0]["data"] is dummy_data
    assert calls[0]["num_workers"] == 4
    assert calls[0]["dialect"] == "duckdb"


def test_store_relational_db_in_neo4j_neo4j_conn_is_optional(monkeypatch):
    """neo4j_conn=None is accepted without error (it is unused by the current implementation)."""
    # Patch get_neo4j_conn before write_to_graph is imported.
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.neo4j.get_neo4j_conn",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_retriever.tabular_data.ingestion.write_to_graph.populate_tabular_data",
        lambda data, num_workers, dialect: None,
    )

    dummy_data = {k: pd.DataFrame() for k in ("tables", "columns", "views", "pks", "fks")}
    store_relational_db_in_neo4j(data=dummy_data)  # neo4j_conn defaults to None
