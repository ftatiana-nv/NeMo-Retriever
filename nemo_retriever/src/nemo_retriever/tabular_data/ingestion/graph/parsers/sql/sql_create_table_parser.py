from nemo_retriever.tabular_data.ingestion.graph.parsers.sql import sql_select_parser as sql_select
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import Props
from nemo_retriever.tabular_data.ingestion.graph.model.query import MissingDataError, Query


def build_query_obj(
    query_obj: Query,
    parsed_query: list[dict[str, str]],
    keep_string_vals: bool,
    schemas: dict[str, any],
    is_full_parse: bool = False,
):
    if len(parsed_query[0]["CreateTable"]["name"]) == 2:
        schema_name = parsed_query[0]["CreateTable"]["name"][0]["value"]
        table_name = parsed_query[0]["CreateTable"]["name"][1]["value"]
    elif len(parsed_query[0]["CreateTable"]["name"]) == 1:
        schema_name = ""
        table_name = parsed_query[0]["CreateTable"]["name"][0]["value"]
    else:
        schema_name = parsed_query[0]["CreateTable"]["name"][1]["value"]
        table_name = parsed_query[0]["CreateTable"]["name"][2]["value"]

    if (query_obj.default_schema is None or len(query_obj.default_schema) == 0) and len(
        schema_name
    ) > 0:
        query_obj.default_schema = schema_name

    is_clone = parsed_query[0]["CreateTable"]["clone"]
    if is_clone is not None:
        raise Exception("Clone is not supported.")

    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)
    if table_node is None:
        raise MissingDataError(
            f"Table {table_name} does not exist in the schema {schema_name}."
        )

    if "query" in parsed_query[0]["CreateTable"]:
        source_sql_query = parsed_query[0]["CreateTable"]["query"]
        if (
            not source_sql_query and len(parsed_query) > 1
        ):  # for ms dialect queries like create->insert
            source_sql_query = parsed_query[1]["Insert"]["source"]
        sql_select.build_query_obj(
            query_obj=query_obj,
            parsed_query=source_sql_query,
            keep_string_vals=keep_string_vals,
            is_full_parse=is_full_parse,
        )
        selected_nodes = query_obj.get_projection_nodes()
    else:
        return False

    for source_col in selected_nodes:
        target_col = query_obj.get_column_node(source_col.get_name(), table_node)
        if target_col.get_match_props()["table_name"] == "":
            raise MissingDataError(
                f"Column {source_col.get_name()} does not exist in the target table {schema_name}.{table_name}."
            )
        edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
        query_obj.get_source_to_target_edges().append(
            (source_col, target_col, edge_params)
        )
    return True
