class Labels:
    COMMAND = "command"
    OPERATOR = "operator"
    FUNCTION = "function"
    CONSTANT = "constant"
    ALIAS = "alias"
    SQL = "sql"
    COLUMN = "column"
    TEMP_COLUMN = "temp_column"
    TABLE = "table"
    TEMP_TABLE = "temp_table"
    SCHEMA = "schema"
    TEMP_SCHEMA = "temp_schema"
    DB = "db"
   

    LIST_OF_ALL = [
        DB,
        SCHEMA,
        TABLE,
        COLUMN,
        TEMP_SCHEMA,
        TEMP_TABLE,
        TEMP_COLUMN,
        SQL,
        COMMAND,
        OPERATOR,
        FUNCTION,
        CONSTANT,
       
       
    ]


labels_to_types = {
    Labels.TABLE: "base table",
}


class Props:
    """Edge/node property keys (used by utils_dal, node)."""
    JOIN = "join"
    SOURCE_SQL_ID = "source_sql_id"
    UNION = "union"
    SQL_ID = "sql_id"


# Labels that have no parent owner in the graph (used by get_entity_before_update).
entities_without_owners = []

# Relationship types for owner traversal (used by get_node_parent_owner_by_id).
data_relationships = ["schema"]






def label_to_type(label: str) -> str:
    if label in labels_to_types:
        return labels_to_types[label]
    return label
