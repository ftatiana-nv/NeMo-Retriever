import sys
import pendulum
import pandas as pd
from nemo_retriever.structured_data.duckdb_engine import DuckDBEngine
from nemo_retriever.relational_db.population.graph.utils import (
    load_fks,
    load_pks,
    load_tables,
    load_columns,
)


def create_dataframe(settings):
    duckdb_connector = DuckDBEngine({"database": "./spider2.duckdb"})

    queries = duckdb_connector.get_queries()

    schema = duckdb_connector.get_schemas()
    tables = schema[0]
    columns = schema[1]

    views = duckdb_connector.get_views()

    pull_df = pd.DataFrame(duckdb_connector.pull_info).explode("schemas")
    pull_df = pull_df.rename(
        {"db_name": "database", "schemas": "schema"}, axis=1
    )

    tables = tables.merge(pull_df)
    columns = columns.merge(pull_df)
    views = views.merge(pull_df)

    pks = duckdb_connector.get_pks()
    fks = duckdb_connector.get_fks()

    return tables, columns, views, queries, pks, fks


def data_for_populate_structured(settings):
    """Build the `data` dict expected by populate_structured_data from create_dataframe output."""
    tables, columns, views, queries, pks, fks = create_dataframe(settings)
    tables = load_tables(tables)
    columns = load_columns(columns)
    pks = load_pks(pks)
    fks = load_fks(fks)
    data = {
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
    }
    # queries is not used by populate_structured_data; include if needed elsewhere
    return data


def extract_relational_db(neo4j_conn=None, params=None):
    """Build data and run populate_structured_data.

    Args:
        neo4j_conn: Active Neo4jConnectionManager instance (passed from the
            orchestrating ingest step; created internally if not provided).
        params: StructuredExtractParams instance.  When provided,
            ``params.db_connection_string`` overrides the default database path.
    """
    import logging
    from nemo_retriever.relational_db.population.populate_data import populate_structured_data

    logger = logging.getLogger(__name__)

    db_path = (
        params.db_connection_string
        if params is not None and params.db_connection_string is not None
        else "./spider2.duckdb"
    )
    settings = {"connection_properties": {"database": db_path}}
    data = data_for_populate_structured(settings)
    populate_structured_data(
        data,
        num_workers=4,
        dialect="duckdb",
    )


if __name__ == "__main__":
    main()