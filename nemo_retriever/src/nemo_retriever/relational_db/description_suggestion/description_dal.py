from enum import Enum

# TODO how to do the connection
from nemo_retriever.relational_db.description_suggestion.functions import create_analysis_name_prompt, create_analysis_prompt, create_query_prompt
from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn
from nemo_retriever.relational_db.population.graph.dal.utils_dal import get_node_properties_by_id
from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels
from pydantic import BaseModel, Field

conn = get_neo4j_conn()


class LabelsWithDescriptionSuggestion(str, Enum):
    TERM = Labels.BT
    ANALYSIS = Labels.ANALYSIS
    ATTRIBUTE = Labels.ATTR
    SNIPPET = "snippet"
    QUERY = Labels.SQL
    SCHEMA = Labels.SCHEMA
    TABLE = Labels.TABLE
    COLUMN = Labels.COLUMN


def get_schema_table_columns_by_table_id(account_id: str, table_id: str):
    query = """MATCH(t:table{id: $table_id, account_id: $account_id})-[:schema]->(c:column)
            RETURN t.name as table_name, t.schema_name as schema_name, collect(c.name) as columns"""
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "table_id": table_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_schema_tables_by_schema_id(account_id: str, schema_id: str):
    try:
        query = """MATCH(db:db{account_id: $account_id})-[:schema]->(s:schema{id: $schema_id})
                OPTIONAL MATCH(s)-[:schema]->(t:table)
                WHERE t IS NOT NULL
                RETURN s.name as schema_name, db.name as db_name, coalesce(collect(t.name), []) as tables"""
        result = conn.query_read_only(
            query,
            parameters={
                "account_id": account_id,
                "schema_id": schema_id,
            },
        )
        if len(result) == 0:
            return None
        # Ensure tables is a list, not [null]
        result_data = result[0]
        if (
            result_data.get("tables")
            and len(result_data["tables"]) == 1
            and result_data["tables"][0] is None
        ):
            result_data["tables"] = []
        return result_data
    except Exception as e:
        logger.error(
            f"Error getting schema tables for schema_id {schema_id}: {e}", exc_info=True
        )
        return None


def get_table_columns_by_column_id(account_id: str, column_id: str):
    query = """match(t:table)-[:schema]->(c:column{id: $column_id, account_id: $account_id})
            OPTIONAL match(t)-[:schema]->(other_column:column)
            where other_column <> c
            return t.name as table_name, c.name as column_name, collect(other_column.name) as columns
            """
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "column_id": column_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_term_name_by_id(account_id: str, term_id: str):
    query = """MATCH(t:term{id: $term_id, account_id: $account_id}) 
               RETURN t.name as name"""
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "term_id": term_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_term_attribute_name_by_attribute_id(account_id: str, attribute_id: str):
    query = """MATCH(t:term{account_id: $account_id})-[:term_of]->(a:attribute{id: $attribute_id}) 
               RETURN t.name as term_name, a.name as attribute_name"""
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "attribute_id": attribute_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_metric_formula_name_by_id(account_id: str, metric_id: str):
    query = """MATCH(m:metric{id: $metric_id, account_id: $account_id}) 
               RETURN m.formula as formula, m.name as name, m.sql as sql"""
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "metric_id": metric_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_analysis_query_name_by_id(account_id: str, analysis_id: str):
    query = """MATCH(a:analysis{id: $analysis_id, account_id:$account_id})-[:analysis_of]->(s:sql) 
               RETURN a.name as name, s.sql_full_query as query_text"""
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "analysis_id": analysis_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_query_text_by_id(account_id: str, query_id: str):
    query = """MATCH(s:sql|analysis{id: $query_id, account_id: $account_id})
                RETURN CASE 
                    WHEN s:sql THEN s.sql_full_query
                    WHEN s:analysis THEN s.sql
                    ELSE NULL
                    END AS query_text
    """
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "query_id": query_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_snippet_query_text_by_id(account_id: str, query_id: str):
    query = """
        MATCH (root:attribute|metric{account_id:$account_id})-[snippet:attr_of|metric_sql{sql_snippet_id: $id}]->()
        RETURN CASE 
            WHEN root:attribute THEN snippet.sql_snippet
            WHEN root:metric THEN snippet.sql
            ELSE NULL
            END AS query_text
    """
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "id": query_id,
        },
    )
    return None if len(result) == 0 else result[0]


def get_all_node_ids_for_label(account_id: str, label: str) -> list[str]:
    """Return all node IDs for a given Neo4j label and account_id (non-deleted only when applicable)."""
    # table/column/schema often have a 'deleted' property
    deleted_filter = ""
    if label in ("table", "column", "schema"):
        deleted_filter = " WHERE coalesce(n.deleted, false) = false"
    query = f"""MATCH (n:{label} {{account_id: $account_id}}){deleted_filter}
                RETURN n.id as id"""
    result = conn.query_read_only(query, parameters={"account_id": account_id})
    return [r["id"] for r in result] if result else []


def get_all_table_ids(account_id: str) -> list[str]:
    """Return all table node IDs for the account (non-deleted tables only)."""
    return get_all_node_ids_for_label(account_id, Labels.TABLE)


def populate_description_suggestions_for_tables(account_id: str) -> list[str]:
    """Generate and insert description suggestions for all tables. Returns list of table IDs that got a suggestion."""
    table_ids = get_all_table_ids(account_id)
    updated = []
    for table_id in table_ids:
        suggestion = get_description_suggestion(account_id, table_id)
        if suggestion is not None:
            updated.append(table_id)
    return updated


def populate_description_suggestions_for_label(
    account_id: str, entity_label: LabelsWithDescriptionSuggestion
) -> list[str]:
    """Generate and insert description suggestions for all nodes of the given type. Returns list of node IDs updated."""
    label_str = str(entity_label.value)
    node_ids = get_all_node_ids_for_label(account_id, label_str)
    updated = []
    for node_id in node_ids:
        suggestion = get_suggestion(account_id, node_id, entity_label)
        if suggestion is not None:
            updated.append(node_id)
    return updated


description_suggestions_map: dict[LabelsWithDescriptionSuggestion, dict] = {
    LabelsWithDescriptionSuggestion.SCHEMA: {
        "get": get_schema_tables_by_schema_id,
        "create_prompt": create_schema_prompt,
    },
    LabelsWithDescriptionSuggestion.TABLE: {
        "get": get_schema_table_columns_by_table_id,
        "create_prompt": create_table_prompt,
    },
    LabelsWithDescriptionSuggestion.COLUMN: {
        "get": get_table_columns_by_column_id,
        "create_prompt": create_column_prompt,
    },
    LabelsWithDescriptionSuggestion.TERM: {
        "get": get_term_name_by_id,
        "create_prompt": create_term_prompt,
    },
    LabelsWithDescriptionSuggestion.ATTRIBUTE: {
        "get": get_term_attribute_name_by_attribute_id,
        "create_prompt": create_attribute_prompt,
    },
    LabelsWithDescriptionSuggestion.METRIC: {
        "get": get_metric_formula_name_by_id,
        "create_prompt": create_metric_prompt,
    },
    LabelsWithDescriptionSuggestion.ANALYSIS: {
        "get": get_analysis_query_name_by_id,
        "create_prompt": create_analysis_prompt,
    },
    LabelsWithDescriptionSuggestion.QUERY: {
        "get": get_query_text_by_id,
        "create_prompt": create_query_prompt,
    },
    LabelsWithDescriptionSuggestion.SNIPPET: {
        "get": get_snippet_query_text_by_id,
        "create_prompt": create_query_prompt,
    },
   
}

name_suggestions_map = {
    LabelsWithDescriptionSuggestion.ANALYSIS: {
        "get": get_query_text_by_id,
        "create_prompt": create_analysis_name_prompt,
    },
}


class NameSuggestionResponse(BaseModel):
    name: str = Field(description="The suggested name for the entity")


class DescriptionSuggestionResponse(BaseModel):
    description: str = Field(description="The suggested description for the entity")


def get_suggestion(
    account_id: str,
    node_id: str,
    entity_label: LabelsWithDescriptionSuggestion,
):
# TODO where is account_metadata coming from?

    # account_metadata = get_account_metadata(account_id)
    account_metadata = {}
    entity_suggestion = description_suggestions_map[entity_label]
    result = entity_suggestion["get"](account_id, node_id)
    prompt, system_prompt = entity_suggestion["create_prompt"](
        result, account_metadata.get("ontology", None)
    )

    # If prompt creation failed, do nothing
    if not prompt or not system_prompt:
        return None

    llm = get_llm_client(kind="langchain")
    if llm is None:
        return None

 # TODO where is SystemMessage and HumanMessage coming from?
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    # TODO where is invoke_with_structured_output coming from?
    response = invoke_with_structured_output(llm, messages, DescriptionSuggestionResponse)
    if not response or not response.description:
        return None
    suggestion = response.description.strip()
    if not suggestion:
        return None

    # Persist in graph
    insert_description_suggestion(account_id, entity_label, node_id, suggestion)
    return suggestion



def get_description_suggestion(account_id: str, node_id: str):
    entities_with_descriptions_suggestions = [
        e.value for e in LabelsWithDescriptionSuggestion
    ]
    entity = get_node_properties_by_id(
        account_id, node_id, entities_with_descriptions_suggestions
    )
    if entity is None:  # snippet is a link
        entity = get_snippet_by_id(account_id, node_id)

    entity_label_str = entity["label"]

    if "description_suggestion" in entity and entity["description_suggestion"]:
        return entity["description_suggestion"]
    try:
        entity_label = LabelsWithDescriptionSuggestion(entity_label_str)
    except ValueError:
        return None

    suggestion = get_suggestion(account_id, node_id, entity_label)
    return suggestion


def generate_name_suggestion(
    account_id: str,
    node_id: str,
    entity_type: LabelsWithDescriptionSuggestion,
):
    # account_metadata = get_account_metadata(account_id)
    account_metadata = {}
    name_suggestion = None
    entity_suggestion = name_suggestions_map[entity_type]
    result = entity_suggestion["get"](account_id, node_id)
    if result:
        prompt, system_prompt = entity_suggestion["create_prompt"](
            result, account_metadata.get("ontology", None)
        )
        llm = get_llm_client(kind="langchain")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        response = invoke_with_structured_output(llm, messages, NameSuggestionResponse)
        name_suggestion = response.name if response else None

    return name_suggestion



def insert_description_suggestion(
    account_id: str,
    label: LabelsWithDescriptionSuggestion,
    node_id: str,
    description_suggestion: str,
):
    if label == LabelsWithDescriptionSuggestion.SNIPPET:
        return insert_snippet_description_suggestion(
            account_id, node_id, description_suggestion
        )

    suggestion = description_suggestion.strip().capitalize()
    label_str = str(label.value)
    query = f"""match(n:{label_str} {{account_id: $account_id, id: $node_id}})
                set n.description_suggestion = $description_suggestion"""
    result = conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "node_id": node_id,
            "description_suggestion": suggestion,
        },
    )
    return result


def insert_snippet_description_suggestion(
    account_id: str, snippet_id: str, description_suggestion: str
):
    suggestion = description_suggestion.strip().capitalize()
    query = """
        MATCH (:attribute|metric{account_id:$account_id})-[n:attr_of|metric_sql{sql_snippet_id: $snippet_id}]-() 
        SET n.description_suggestion = $description_suggestion
    """
    return conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "snippet_id": snippet_id,
            "description_suggestion": suggestion,
        },
    )
