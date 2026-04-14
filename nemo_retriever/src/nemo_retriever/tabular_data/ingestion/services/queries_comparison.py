import logging

from pandas import DataFrame
from typing import Callable, Union
from networkx import Graph, DiGraph
from networkx.algorithms.isomorphism import DiGraphMatcher

from nemo_retriever.tabular_data.ingestion.graph.dal.queries_dal import get_sql_by_id
from nemo_retriever.tabular_data.ingestion.graph.model.query import Query
from nemo_retriever.tabular_data.ingestion.graph.model.node import Node
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import SQLType, Labels


logger = logging.getLogger(__name__)


def compare_query_to_list_of_queries(
    main_graph: Graph,
    queries_ids: list[str],
    get_sql_string_by_id: Callable,
    get_parsed_query: Callable[[str], Query],
    save_leafs_in_df: Callable = None,
    is_subgraph: bool = False,
    remove_aliases: bool = False,
):
    ids_of_matching_sqls = []
    ids_of_subset_sqls = []
    for sql_id in queries_ids:
        compared_sql = get_sql_string_by_id(sql_id)
        try:
            compared_query = get_parsed_query(compared_sql)
            compared_graph = get_slim_graph_from_edges(
                compared_query.get_edges(),
                remove_aliases=remove_aliases,
            )
            if save_leafs_in_df is not None:
                save_leafs_in_df(sql_id, compared_graph)
            is_equal = compare_queries(
                main_graph, compared_graph, is_subgraph=is_subgraph
            )
        except Exception as err:
            logger.exception(err)
            logger.info("Failed comparing.")
            is_equal = False

        if is_equal:
            ## heuristic: if a query is a subgraph of another query and the nodes count is equal- then it's identical
            if is_subgraph:
                if (
                    main_graph.number_of_nodes() != compared_graph.number_of_nodes()
                    or main_graph.number_of_edges() != compared_graph.number_of_edges()
                ):
                    ids_of_subset_sqls.append(sql_id)
                else:
                    ids_of_matching_sqls.append(sql_id)
            else:
                ids_of_matching_sqls.append(sql_id)
                return ids_of_matching_sqls, ids_of_subset_sqls
    return ids_of_matching_sqls, ids_of_subset_sqls


def edge_compare(
    edge1: dict[str, str | None],
    edge2: dict[str, str | None],
):
    if (
        "child_idx" in edge1
        and edge1["child_idx"] is not None
        and "child_idx" in edge2
        and edge2["child_idx"] is not None
    ):
        return edge1["child_idx"] == edge2["child_idx"]
    return True


def node_compare(node1: dict[str, str], node2: dict[str, str]):
    if node1["label"] != node2["label"]:
        return False
    if node1["label"] == Labels.SQL:
        return True
    if node1["label"] in [Labels.TABLE, Labels.COLUMN]:
        return node1["id"] == node2["id"]
    return node1["name"] == node2["name"]


def get_set_of_edges(graph: Graph):
    edges = set()
    for edge in graph.edges.data():
        try:
            edge_label = edge[2].get("child_idx")
            edges.add(
                graph.nodes[edge[0]]["node_label"].lower()
                + "_"
                + graph.nodes[edge[1]]["node_label"].lower()
                + "_"
                + (f"{edge_label}" if edge_label else "")
            )
        except Exception as e:
            print(f"Error getting edge label for edge {edge}: {e}")
    return edges


def compare_sets_of_edges_by_order(
    main_set: set, compared_set: set, is_full_comparison=False
):
    if is_full_comparison and len(main_set) != len(compared_set):
        return False

    for edge in main_set:
        if edge not in compared_set:
            return False
    return True


def compare_queries(
    main_graph: Graph, compared_graph: Graph, is_subgraph=False
) -> bool:
    if is_subgraph:
        ## subgraph isomorphism
        return DiGraphMatcher(
            G1=compared_graph,  # Check whether a subgraph of G1 is isomorphic to G2
            G2=main_graph,
            node_match=node_compare,
            edge_match=edge_compare,
        ).subgraph_is_isomorphic()
    # full comparison
    main_set = get_set_of_edges(main_graph)
    compared_set = get_set_of_edges(compared_graph)
    return compare_sets_of_edges_by_order(
        main_set, compared_set, is_full_comparison=True
    )


aliases_labels = [Labels.ALIAS, Labels.SET_OP_COLUMN]


def get_node_label(node):
    if node.label == Labels.SQL:
        return node.label
    node_label = f"{node.label}_{node.name}"
    if node.label == Labels.TABLE:
        identity_props = node.match_props
        return f"{identity_props['db_name']}.{identity_props['schema_name']}.{identity_props['name']}"
    if node.label == Labels.COLUMN:
        identity_props = node.match_props
        return f"{identity_props['db_name']}.{identity_props['schema_name']}.{identity_props['table_name']}.{identity_props['name']}"
    return node_label


def get_slim_graph_from_edges(
    edges: list[tuple[Node, Node, dict, str]], remove_aliases=False
):
    g = DiGraph()
    aliases_ids = set()
    for e in edges:
        node_e0 = e[0].get_sql_node() if isinstance(e[0], Query) else e[0]
        node_e1 = e[1].get_sql_node() if isinstance(e[1], Query) else e[1]
        if node_e0.label in aliases_labels:
            aliases_ids.add(node_e0.id)
        if node_e1.label in aliases_labels:
            aliases_ids.add(node_e1.id)
        g.add_nodes_from(
            [
                (
                    node_e0.id,
                    {
                        "name": node_e0.name,
                        "label": node_e0.label,
                        "id": node_e0.id,
                        "node_label": get_node_label(node_e0),
                    },
                ),
                (
                    node_e1.id,
                    {
                        "name": node_e1.name,
                        "label": node_e1.label,
                        "id": node_e1.id,
                        "node_label": get_node_label(node_e1),
                    },
                ),
            ]
        )
        child_idx = None if not e[2] or "child_idx" not in e[2] else e[2]["child_idx"]
        g.add_edges_from([(node_e0.id, node_e1.id)], child_idx=child_idx)
    if remove_aliases:
        g = remove_aliases_from_graph(g, aliases_ids)
    return g


def remove_aliases_from_graph(g: DiGraph, aliases_ids: set[str]):
    removed_aliases_ids = []
    for alias_id in aliases_ids:
        for predecessor in g.predecessors(alias_id):
            for successor in g.successors(alias_id):
                g.add_edge(predecessor, successor)
        try:
            g.remove_node(alias_id)
            removed_aliases_ids.append(alias_id)
        except Exception:
            pass
    return g


def get_sqls_connected_to_tables(
    sql_type: SQLType,
    tbl_ids: list[str],
    nodes_count: int,
    sqls_tbls_df: DataFrame,
    parsed_query_leaves: list[tuple[str, str]],
    is_subgraph: bool = False,
) -> list[str]:
    if sqls_tbls_df.empty:
        return []

    # s.id as sql_id, collect(distinct node.is) as tbls, s.nodes_count as nodes_count
    if not is_subgraph:
        potential_sqls_list = list(
            sqls_tbls_df.loc[
                (sqls_tbls_df.sql_type == sql_type)
                & (sqls_tbls_df.nodes_count == nodes_count)
                & sqls_tbls_df.apply(lambda x: set(x.tbls) == set(tbl_ids), axis=1)
                & sqls_tbls_df.apply(
                    lambda x: x.leaves is None
                    or set(x.leaves) == set(parsed_query_leaves),
                    axis=1,
                )
            ].sql_id
        )
    else:  # is subgraph
        potential_sqls_list = list(
            sqls_tbls_df.loc[
                (sqls_tbls_df.nodes_count >= nodes_count)
                & sqls_tbls_df.apply(
                    lambda x: set(x.tbls).issubset(set(tbl_ids)), axis=1
                )
                & sqls_tbls_df.apply(
                    lambda x: x.leaves is None
                    or set(x.leaves).issubset(set(parsed_query_leaves)),
                    axis=1,
                )
            ].sql_id
        )
    return potential_sqls_list


def get_leafs_from_graph(query_slim_graph: DiGraph):
    return [
        (
            query_slim_graph.nodes[node_id]["label"],
            query_slim_graph.nodes[node_id]["name"],
        )
        for node_id in query_slim_graph.nodes()
        if query_slim_graph.out_degree(node_id) == 0
    ]


def find_identical_queries(
    get_parsed_query: Callable[[str], Query],
    main_sql: str,
    sqls_tbls_from_graph_df: Union[DataFrame, None] = None,
    is_subgraph: bool = False,
    remove_aliases: bool = False,
    in_memory_queries: Union[dict[str, Query], None] = None,
    potential_sqls_ids: Union[list[str], None] = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if sqls_tbls_from_graph_df is None and potential_sqls_ids is None:
        raise ValueError(
            "sqls_tbls_from_graph_df or potential_sqls_ids must be provided"
        )
    parsed_query_obj = get_parsed_query(main_sql)
    main_graph = get_slim_graph_from_edges(
        parsed_query_obj.get_edges(),
        remove_aliases=remove_aliases,
    )
    parsed_query_leaves = get_leafs_from_graph(main_graph)

    if potential_sqls_ids is None:
        potential_sqls_ids = get_sqls_connected_to_tables(
            parsed_query_obj.sql_node.props["sql_type"],
            parsed_query_obj.get_tables_ids(),
            parsed_query_obj.get_nodes_counter(),
            sqls_tbls_from_graph_df,
            parsed_query_leaves,
            is_subgraph,
        )

    def save_leafs_in_df(query_id: str, query_slim_graph: DiGraph):
        if sqls_tbls_from_graph_df is not None:
            saved_query_leaves = sqls_tbls_from_graph_df.loc[
                sqls_tbls_from_graph_df.sql_id == query_id, "leaves"
            ]
            if list(saved_query_leaves)[0] is None:
                query_leafs = get_leafs_from_graph(query_slim_graph)
                mask = sqls_tbls_from_graph_df.sql_id == query_id
                row_idx = sqls_tbls_from_graph_df.index[mask][0]
                sqls_tbls_from_graph_df.at[row_idx, "leaves"] = query_leafs

    identical_sqls_ids_from_graph = []
    subset_ids_from_graph = []
    if potential_sqls_ids:
        ## search in graph
        identical_sqls_ids_from_graph, subset_ids_from_graph = (
            compare_query_to_list_of_queries(
                main_graph=main_graph,
                queries_ids=potential_sqls_ids,
                get_sql_string_by_id=lambda id: get_sql_by_id(id),
                get_parsed_query=get_parsed_query,
                save_leafs_in_df=save_leafs_in_df,
                remove_aliases=remove_aliases,
                is_subgraph=is_subgraph,
            )
        )
    identical_sqls_ids_in_memory = []
    subset_ids_from_memory = []
    # we get in_memory_queries dictionary during population
    if len(identical_sqls_ids_from_graph) == 0 and in_memory_queries:
        identical_sqls_ids_in_memory, subset_ids_from_memory = (
            compare_query_to_list_of_queries(
                main_graph=main_graph,
                queries_ids=in_memory_queries.keys(),
                get_parsed_query=get_parsed_query,
                get_sql_string_by_id=lambda id: in_memory_queries[id].string_query,
                is_subgraph=is_subgraph,
                remove_aliases=remove_aliases,
            )
        )
    # when we're checking full equality we expect to have only one identical sql id (either from graph or in memory)
    return (
        identical_sqls_ids_from_graph,
        identical_sqls_ids_in_memory,
        subset_ids_from_graph,
        subset_ids_from_memory,
    )
