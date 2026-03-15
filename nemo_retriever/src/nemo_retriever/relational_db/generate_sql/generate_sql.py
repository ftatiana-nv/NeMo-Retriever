from nemo_retriever.relational_db.sql_tool.generate_sql import get_sql_tool_response


def generate_sql(query: str) -> str:
    """Generate a SQL query for a given natural language query."""
    return get_sql_tool_response(query)
