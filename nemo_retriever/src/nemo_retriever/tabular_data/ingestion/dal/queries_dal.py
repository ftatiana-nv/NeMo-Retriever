import pandas as pd
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import SQLType
from nemo_retriever.tabular_data.ingestion.graph.model.node import Node
from nemo_retriever.tabular_data.ingestion.graph.model.query import get_sql_counters
from nemo_retriever.tabular_data.ingestion.graph.utils import chunks
from nemo_retriever.tabular_data.ingestion.graph.dal.utils_dal import prepare_edge, add_edges
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

conn = get_neo4j_conn()

def get_sql_by_id(sql_id: str):
    query = "MATCH(n:sql{id:$sql_id}) return n.sql_full_query as sql_full_query"
    result = conn.query_read_only(
        query=query, parameters={"sql_id": sql_id}
    )
    return result[0]["sql_full_query"]


def load_sqls_to_tables(is_view: bool = False):
    query = """
            MATCH (s:sql {is_sub_select:false} WHERE ($is_view=TRUE AND s.sql_type="view") OR ($is_view=FALSE AND s.sql_type<>"view"))
            WITH s
            CALL apoc.path.subgraphNodes(s, {
            relationshipFilter: "SQL>",
            labelFilter: "/table|-column",
            minLevel: 0}) 
            YIELD node
            WITH s.id as sql_id, s.sql_type as sql_type, collect(distinct node.id) as tbls, s.nodes_count as nodes_count
            RETURN collect({sql_id:sql_id, sql_type:sql_type, tbls:tbls, nodes_count:nodes_count, leaves: NULL}) as sqls_tbls
            """
    sqls_tbls_df = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={
                "is_view": is_view,
                "sql_type": SQLType.VIEW,
            },
        )[0]["sqls_tbls"]
    )
    return sqls_tbls_df

def update_counters_and_timestamps_for_query_and_affected_data(
    identical_sql_id: str,
    sql_node: Node,
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
    cypher_query = f"""MATCH (s:sql {{id: $id}}) 
                                SET s.last_query_timestamp = $latest_timestamp 
                                SET s.total_counter = s.total_counter + $total_counter
                                {set_cnts_str}
                            """.strip()
    conn.query_write(
        query=cypher_query,
        parameters={
            "latest_timestamp": latest_timestamp,
            "id": identical_sql_id,
            "total_counter": total_counter,
        },
    )

    if update_data_last_query_timestamp:
        cypher_query = """
                MATCH (v: sql{id:$id})
                WITH v
                CALL apoc.path.subgraphNodes(v, {
                    relationshipFilter: "SQL>", 
                    labelFilter: "/column|/table"})
                    YIELD node
                    WHERE coalesce(node.deleted, false) = false 
                    SET node.last_query_timestamp = $latest_timestamp
                """
        conn.query_write(
            cypher_query,
            parameters={
                "latest_timestamp": latest_timestamp,
                "id": identical_sql_id,
            },
        )

def add_query(edges):
    """
    Add the nodes and edges of the parsed query to the graph
    :param edges: edges ready for insertion
    :return:
    """
    edges_data = [prepare_edge(edge) for edge in edges]
    edges_data_chunks = list(chunks(edges_data, 10))
    for i, chunk in enumerate(edges_data_chunks):
        add_edges(chunk)

def get_sql_by_full_query(sql_full_query: str):
    query = "MATCH(n:sql{sql_full_query:$sql_full_query}) return n.id as id"
    result = conn.query_read_only(
        query=query,
        parameters={"sql_full_query": sql_full_query},
    )
    if result:
        return result[0]["id"]
    return None
