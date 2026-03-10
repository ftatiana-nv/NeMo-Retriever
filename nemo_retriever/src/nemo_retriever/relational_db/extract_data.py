import sys
import pendulum
import pandas as pd
from nemo_retriever.structured_data.duckdb_engine import DuckDBEngine


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


def parse_param(argv):
    ret = {}
    ret["data_interval_start"] = pendulum.parse(argv[1])
    ret["data_interval_end"] = pendulum.parse(argv[2])
    ret["connection_properties"] = argv[3]
    return ret



# settings = parse_param(sys.argv)
# settings = {"data_interval_start": pendulum.parse("2026-03-01"), "data_interval_end": pendulum.parse("2026-03-02"), "connection_properties": {"database": "./spider2.duckdb"}}
# create_dataframe(settings)
