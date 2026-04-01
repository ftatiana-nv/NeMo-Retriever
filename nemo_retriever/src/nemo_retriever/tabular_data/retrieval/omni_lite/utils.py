import logging
from enum import StrEnum
from itertools import groupby


logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

# Hard ceiling on how many candidate snippets we want to reason over for a single question.
# Larger numbers tend to confuse the LLM and increase latency.
MAX_CALCULATION_CANDIDATES = 15

# k for each omni_semantic_candidates call (qnv, qwv, per-entity).
SEMANTIC_CANDIDATES_K = 10

# Per-attempt batch size when pulling semantic candidates. We keep this modest
# to balance recall with response time and to leave room for deduping.
ATTR_CANDIDATE_BATCH_SIZE = 6

# Maximum number of batch pulls before we give up looking for fresh attributes;
# empirically, three rounds either fill the quota or show the backend is stuck on duplicates.
ATTR_CANDIDATE_MAX_ATTEMPTS = 3

# Budget for entity-specific candidate searches. This limits how many candidates we fetch
# per entity when ensuring entity coverage. Distributed evenly across entities (at least 1 per entity).
# This prevents over-fetching when there are many entities while ensuring each gets coverage.
ENTITY_CANDIDATE_BUDGET = 4


QUERIES_USAGE_PERCENTILE = "queries_usage_percentile"
TABLES_USAGE_PERCENTILE = "tables_usage_percentile"
COLUMNS_USAGE_PERCENTILE = "columns_usage_percentile"




class Labels(StrEnum):
    """Semantic labels used by omni-lite candidate retrieval."""

    CUSTOM_ANALYSIS = "custom_analysis"
    COLUMN = "column"

def store_usage_percentiles(
    percentiles_type_name: str,
    usage_percentile_25: int,
    usage_percentile_75: int,
):
    query = """
            MATCH (n:db)
            WITH n
            CALL apoc.create.setProperties(n, [$percentiles_type_name_25, $percentiles_type_name_75], [$usage_percentile_25, $usage_percentile_75]) 
            YIELD node
            RETURN n
            """
    conn.query_write(
        query=query,
        parameters={
            "percentiles_type_name_25": f"{percentiles_type_name}_25",
            "percentiles_type_name_75": f"{percentiles_type_name}_75",
            "usage_percentile_25": usage_percentile_25,
            "usage_percentile_75": usage_percentile_75,
        },
    )



def get_stored_usage_percentiles(percentiles_type_name: str):
    query = f"""
                MATCH (n:db)
                RETURN n.{f"{percentiles_type_name}_25"} as usage_percentile_25, n.{f"{percentiles_type_name}_75"} as usage_percentile_75
                """
    results = conn.query_read_only(
        query=query,
        parameters={
        },
    )
    return results

def get_count_str_by_month(alias: str):
    current = date.today().replace(day=1)
    count_3_month = []

    for i in range(0, 3):
        count_3_month.append(f"coalesce({alias}.cnt_{current.month}_{current.year}, 0)")
        prev = current - timedelta(days=1)
        current = prev.replace(day=1)

    count_str = "+".join(count_3_month)
    return count_str

def init_queries_usage_percentiles():
    count_string = get_count_str_by_month("n")
    query_all = f"""match(n:sql{{is_sub_select:FALSE}})
                     return collect({count_string}) as usages """
    usages_result = conn.query_read_only(
        query=query_all, parameters={}
    )
    usages = usages_result[0]["usages"]
    if len(usages) == 0:
        return 0, 0
    usage_percentile_25 = np.percentile(usages, 25)
    usage_percentile_75 = np.percentile(usages, 75)
    store_usage_percentiles(
        QUERIES_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75
    )
    return usage_percentile_25, usage_percentile_75


def get_usage_percentiles():
    stored_percentiles = get_stored_usage_percentiles(
        QUERIES_USAGE_PERCENTILE
    )
    if len(stored_percentiles) == 0 or (
        stored_percentiles[0]["usage_percentile_25"] is None
    ):
        usage_percentile_25, usage_percentile_75 = init_queries_usage_percentiles()
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75


def get_queries_usage_percentiles(node_str="node"):
    count_str = get_count_str_by_month(node_str)
    usage_percentile_25, usage_percentile_75 = get_usage_percentiles()
    return usage_percentile_25, usage_percentile_75, count_str

def expand_info(ids_and_labels):
    # define a function for key
    def key_func(k):
        return k["label"]

    results = {}

    (
        queries_percentile_25,
        queries_percentile_75,
        cnt_str,
    ) = get_queries_usage_percentiles("sql_node")

    for label, ids in groupby(ids_and_labels, key_func):
        label_id_pairs_for_current_label = list(ids)
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id:label_id.id}})
                    CALL apoc.case([
                        n:term,
                            'MATCH (n)-[:term_of]->(attr:attribute)
                            WITH n, collect(properties(attr)) as attributes
                            RETURN apoc.map.setKey(properties(n), "attributes", attributes) as item',
                        n:attribute,
                            'MATCH(term:term)-[:term_of]->(n)
                             CALL {{
                                WITH n
                                MATCH (n)-[snippet:attr_of]->(item:column|error_node|sql|alias)
                                WITH n, snippet, item,
                                     CASE WHEN item:sql OR item:alias THEN TRUE ELSE FALSE END AS complex_snippet
                                CALL apoc.case([
                                        item:column, "match(item)<-[:schema]-(table:table) {get_queries_for_column("table", "item", cnt_str)} return  usage",
                                        item:sql or item:alias , "{get_sql_or_alias_roots(cnt_str)}"
                                        ],
                                    "return 0 AS usage",
                                    {{item:item, account_id:$account_id, usage_percentile_25: $usage_percentile_25,
                                      usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }})
                                    YIELD value as usage
                                WITH n, snippet, usage.usage as usage, complex_snippet
                                RETURN collect(distinct {{sql_code: snippet.sql_snippet, snippet_id: snippet.sql_snippet_id, usage: usage}}) as sql,
                                       reduce(res=FALSE, flag IN collect(complex_snippet) |res OR flag) as complex_attribute
                                UNION
                                WITH n
                                OPTIONAL MATCH (n)-[:attr_of]->(dummy:column|error_node|sql|alias)
                                WITH n, count(dummy) as snippet_count
                                WHERE snippet_count = 0
                                RETURN [] as sql, false as complex_attribute
                             }}
                             RETURN apoc.map.setPairs(properties(n),[["owner_id", term.owner_id],["term_name", term.name], ["sql", sql], ["complex_attribute", complex_attribute]]) as item',
                        n:analysis,
                            'MATCH(n)-[:analysis_of]->(sql:sql)
                            WITH n, collect(distinct {{sql_code: sql.sql_full_query}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:metric,
                            'OPTIONAL MATCH(n)-[metric_sql:metric_sql]->(attribute:attribute)
                            WITH n, collect(distinct {{sql_code: metric_sql.sql}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:field,
                            'MATCH(n)<-[:{"|".join(fields_relationships)}]-(parent) 
                            RETURN apoc.map.setPairs(properties(n),[["table_name", parent.name],["table_type", parent.type], ["parent_id", parent.id]]) as item',
                        n:column,
                            'MATCH(n)<-[:schema]-(parent) 
                            RETURN apoc.map.setPairs(properties(n),[["table_name", parent.name],["table_type", parent.type], ["parent_id", parent.id]]) as item'
                        ],
                        
                        'with n RETURN n{{ .*}} as item ',
                        {{n:n, account_id:$account_id, sql_type: $sql_type, usage_percentile_25: $usage_percentile_25,
                        usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "label_id_pairs": label_id_pairs_for_current_label,
            "sql_type": SQLType.QUERY,
            "usage_percentile_25": queries_percentile_25,
            "usage_percentile_75": queries_percentile_75,
        }
        params.update(queries_for_columns_params)
        result = conn.query_read_only(
            query=query,
            parameters=params,
        )
        if len(result) > 0:
            # Add additional info
            for id, item in result[0]["ids_to_props"].items():
                if item["type"] == Labels.BT:
                    term_tables = get_term_tables(
                        account_id, user_participants, item["id"], 0, 100
                    )
                    item["tables"] = term_tables
                elif item["type"] == Labels.ATTR:
                    attr_columns = get_attr_columns(
                        account_id, user_participants, item["id"], 0, 100
                    )
                    item["columns"] = attr_columns

            results = results | result[0]["ids_to_props"]

    return results


def get_semantic_candidates_information(
    entity: str,
    k: int = 5,
    threshold: float = 0,
    list_of_semantic: list = Labels.LIST_OF_SEMANTIC,
):
    labels = list_of_semantic
    results = []
    

    # TODO: use Retriever class to search semantic vector index
    # nodes_results = search_vector_index(
    #     "nodes_vector_store",
    #     allowed_ids,
    #     entity,
    #     k=k,
    #     label_filter=labels,
    # )
    # results.extend([item["metadata"] for item in nodes_results])

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    candidates_properties = expand_info(ids_and_labels)
    for c in results:
        update_candidate_properties(account_id, c, candidates_properties)

    # score is similarity, higher is better
    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results








def _semantic_pull(query: str, list_of_semantic: list[str]) -> list[dict]:
    text = (query or "").strip()
    if not text:
        return []
    return (
        get_semantic_candidates_information(
            text,
            k=SEMANTIC_CANDIDATES_K,
            list_of_semantic=list_of_semantic,
        )
        or []
    )


def _dedupe_best_score_sort_cap(combined: list[dict]) -> list[dict]:
    """Deduplicate by (label, id), keep best score, sort descending, cap."""
    best_by_key: dict[tuple[str | None, str], dict] = {}
    for c in combined:
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        score = float(c.get("score") or 0)
        prev = best_by_key.get(key)
        if prev is None or score > float(prev.get("score") or 0):
            best_by_key[key] = c

    unique = list(best_by_key.values())
    unique.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return unique[:MAX_CALCULATION_CANDIDATES]


def extract_candidates(
    entities: list[str],
    query_no_values: str,
    query_with_values: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    One semantic search per string: ``query_no_values``, ``query_with_values`` (if distinct),
    and each entity name. For each string, pull custom analyses and columns via
    ``omni_semantic_candidates``. Merge streams, dedupe by (label, id) keeping best score,
    sort by score, cap at ``MAX_CALCULATION_CANDIDATES`` per stream.

    Returns:
        ``(custom_analysis_candidates, column_candidates)``
    """
    qnv_text = (query_no_values or "").strip()
    qwv_text = (query_with_values or "").strip()

    pulls: list[str] = []
    if qnv_text:
        pulls.append(qnv_text)
    if qwv_text and qwv_text != qnv_text:
        pulls.append(qwv_text)
    for ent in entities or []:
        t = (ent or "").strip()
        if t:
            pulls.append(t)

    combined_custom: list[dict] = []
    combined_columns: list[dict] = []
    for text in pulls:
        combined_custom.extend(_semantic_pull(text, [Labels.CUSTOM_ANALYSIS]))
        combined_columns.extend(_semantic_pull(text, [Labels.COLUMN]))

    out_custom = _dedupe_best_score_sort_cap(combined_custom)
    out_columns = _dedupe_best_score_sort_cap(combined_columns)

    logger.info(
        f"extract_candidates: {len(out_custom)} custom_analysis, {len(out_columns)} column "
        f"(max {MAX_CALCULATION_CANDIDATES} each), {len(pulls)} pulls"
    )

    return out_custom, out_columns
