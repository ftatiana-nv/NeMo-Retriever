import sys
import pendulum
import pandas as pd
from nemo_retriever.structured_data.duckdb_engine import DuckDBEngine
from nemo_retriever.relational_db.population.graph.utils import (
    load_fks,
    load_pks,
    load_tables,
    load_columns,
    load_views,
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
    tables = load_tables(tables, is_csv=False)
    columns = load_columns(columns, is_csv=False)
    views = load_views(views, is_csv=False)
    pks = load_pks(pks, is_csv=False)
    fks = load_fks(fks, is_csv=False)
    data = {
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
    }
    # queries is not used by populate_structured_data; include if needed elsewhere
    return data


def parse_param(argv):
    ret = {}
    ret["data_interval_start"] = pendulum.parse(argv[1])
    ret["data_interval_end"] = pendulum.parse(argv[2])
    ret["connection_properties"] = argv[3]
    return ret



# Example: build data for populate_structured_data and call it
from nemo_retriever.relational_db.population.populate_data import populate_structured_data

settings = {"connection_properties": {"database": "./spider2.duckdb"}}
data = data_for_populate_structured(settings)
populate_structured_data(
    data,
    account_id="your_account_id",
    num_workers=4,
    dialect="duckdb",
    keep_string_values=False,
)
print("done")
