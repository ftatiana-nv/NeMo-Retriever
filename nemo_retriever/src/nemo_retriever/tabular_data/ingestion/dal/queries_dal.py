# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.tabular_data.ingestion.utils import chunks
from nemo_retriever.tabular_data.ingestion.dal.utils_dal import prepare_edge, add_edges
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode


def add_query(edges):
    """Add the nodes and edges of the parsed query to the graph."""
    edges_data = [prepare_edge(edge) for edge in edges]
    for chunk in chunks(edges_data, 10):
        add_edges(chunk)


def get_sql_by_full_query(sql_full_query: str):
    query = (
        f"MATCH (n:{Labels.SQL} {{sql_full_query: $sql_full_query}}) RETURN n.id AS id"
    )
    result = get_neo4j_conn().query_read(
        query=query,
        parameters={"sql_full_query": sql_full_query},
    )
    if result:
        return result[0]["id"]
    return None


def get_sql_counters(sql_node):
    total_counter = sql_node.get_properties()["total_counter"]
    cnt_per_month = {}
    for prop_key, prop_val in sql_node.get_properties().items():
        if prop_key.startswith("cnt_"):
            cnt_per_month.update({prop_key: prop_val})
    return total_counter, cnt_per_month


def update_counters_and_timestamps_for_query_and_affected_data(
    identical_sql_id: str,
    sql_node: Neo4jNode,
    update_data_last_query_timestamp: bool = True,
):
    # If sql is already in graph and sql is not temporary metric sql - add +1 to sql's counter
    # sql_node = edges[0][0]
    latest_timestamp = sql_node.get_properties()["last_query_timestamp"]
    total_counter, cnt_per_month = get_sql_counters(sql_node)
    set_cnts_str = ""
    for month, cnt in cnt_per_month.items():
        set_cnts_str = f"{set_cnts_str}SET s.{month} = coalesce(s.{month}, 0) + {cnt}\n"
    # if the sql already exists in the graph, then update the "last_query_timestamp" property
    # of the sql_node and the table and column nodes that appear as part of the sql.
    cypher_query = f"""MATCH (s:{Labels.SQL} {{id: $id}})
                                SET s.last_query_timestamp = $latest_timestamp
                                SET s.total_counter = s.total_counter + $total_counter
                                {set_cnts_str}
                            """.strip()
    get_neo4j_conn().query_write(
        query=cypher_query,
        parameters={
            "latest_timestamp": latest_timestamp,
            "id": identical_sql_id,
            "total_counter": total_counter,
        },
    )

    if update_data_last_query_timestamp:
        cypher_query = f"""
                MATCH (v:{Labels.SQL} {{id:$id}})
                WITH v
                CALL apoc.path.subgraphNodes(v, {{
                    relationshipFilter: "SQL>",
                    labelFilter: "/{Labels.COLUMN}|/{Labels.TABLE}"}})
                    YIELD node
                    WHERE coalesce(node.deleted, false) = false
                    SET node.last_query_timestamp = $latest_timestamp
                """
        get_neo4j_conn().query_write(
            cypher_query,
            parameters={
                "latest_timestamp": latest_timestamp,
                "id": identical_sql_id,
            },
        )
