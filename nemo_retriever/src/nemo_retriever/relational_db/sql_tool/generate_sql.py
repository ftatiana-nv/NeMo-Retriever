import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sqlalchemy import create_engine, inspect
from deepagents import create_deep_agent 
from deepagents.backends import FilesystemBackend  
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def _get_sql_agent(base_dir: str):
    """
    Create and cache the Deep Agent with all Snowflake schema tools.
    This runs only once per process; subsequent calls reuse the same agent,
    avoiding repeated schema discovery and tool construction for every question.
    """
    global _sql_agent
    if _sql_agent is not None:
        return _sql_agent

    # Connect to DuckDB database
    duckdb_path = os.environ.get("DUCKDB_PATH", "./spider2.duckdb")
    engine = create_engine(f"duckdb:///{duckdb_path}")
    inspector = inspect(engine)

    # Get all schemas (excluding system schemas)
    all_schemas = inspector.get_schema_names()
    user_schemas = [s for s in all_schemas if s not in ["INFORMATION_SCHEMA"]]

    # Step 2: Collect all tables from all schemas with fully qualified names
    tables_by_schema = {}
    all_table_count = 0
    for schema in user_schemas:
        try:
            tables = inspector.get_table_names(schema=schema)
            if tables:
                tables_by_schema[schema] = tables
                all_table_count += len(tables)
        except Exception as e:
            print(f"Warning: Could not access schema {schema}: {e}")

    print(f"Discovered {all_table_count} tables across {len(user_schemas)} schemas")

    # Step 3: Create multiple SQLDatabase instances (one per schema) in parallel
    llm = ChatNVIDIA(
        base_url=os.environ.get("LLM_INVOKE_URL"),
        api_key=os.environ.get("LLM_API_KEY"),
        model=os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct"),
    )
    all_sql_tools = []

    def process_schema(schema):
        """Process a single schema and return its tools."""
        try:
            # DuckDB uses the shared engine; filter to the current schema
            db_for_schema = SQLDatabase(
                engine,
                schema=schema,
                sample_rows_in_table_info=3,
                view_support=True,
            )

            # Create toolkit and get tools for this schema
            toolkit = SQLDatabaseToolkit(db=db_for_schema, llm=llm)
            schema_tools = toolkit.get_tools()

            # Add schema prefix to tool names for clarity
            for tool in schema_tools:
                if hasattr(tool, "name"):
                    tool.name = f"{schema}_{tool.name}"
                if hasattr(tool, "description"):
                    tool.description = f"[{schema} schema] {tool.description}"

            print(f"Added tools for schema: {schema}")
            return schema_tools

        except Exception as e:
            print(f"Warning: Could not create tools for schema {schema}: {e}")
            return []

    # Process all schemas in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(10, len(user_schemas))) as executor:
        # Submit all schema processing tasks
        future_to_schema = {executor.submit(process_schema, schema): schema for schema in user_schemas}
        
        # Collect results as they complete
        for future in as_completed(future_to_schema):
            schema_tools = future.result()
            all_sql_tools.extend(schema_tools)

    print(f"Total SQL tools created: {len(all_sql_tools)}")

    # Step 4: Create the Deep Agent with all schema tools
    # Only load the core operational skills (schema-exploration, query-writing).
    # Final formatting is handled directly in query-writing (JSON {sql_code, answer, result}).
    _sql_agent = create_deep_agent(
        model=llm,
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/schema-exploration",
            "./skills/query-writing",
        ],
        tools=all_sql_tools,  # SQL database tools from all schemas
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir),  # Persistent file storage
    )

    return _sql_agent


def _parse_markdown_answer(text: str) -> dict | None:
    """
    Best-effort parser for markdown-style answers of the form:

    ### Final Answer

    **SQL Code Executed:**
    ```sql
    SELECT ...
    ```

    **Result:**
    - ...

    **Answer:**
    some explanation...
    """
    sql_code = None
    answer = None
    result_value: object | None = None

    # 1) Extract SQL between ```sql and next ```
    start = text.find("```sql")
    if start != -1:
        start = text.find("\n", start)
        if start != -1:
            end = text.find("```", start)
            if end != -1:
                sql_code_block = text[start:end]
                sql_code = sql_code_block.strip()

    # 2) Extract answer text after "**Answer:**" (if present)
    answer_marker = "**Answer:**"
    idx = text.find(answer_marker)
    if idx != -1:
        answer = text[idx + len(answer_marker) :].strip()

    # 3) Extract a numeric result from the "Result" section if present
    #    We look between "**Result:**" and "**Answer:**" (or end of text)
    result_marker = "**Result:**"
    r_idx = text.find(result_marker)
    if r_idx != -1:
        r_start = r_idx + len(result_marker)
        r_end = text.find("**Answer:**", r_start)
        if r_end == -1:
            r_end = len(text)
        result_section = text[r_start:r_end]
        # Try to find a number in the result section
        import re

        m = re.search(r"-?\d+(\.\d+)?", result_section)
        if m:
            num_str = m.group(0)
            try:
                result_value = float(num_str)
            except ValueError:
                result_value = num_str
        else:
            # Fallback: keep the raw section as result if non-empty
            if result_section.strip():
                result_value = result_section.strip()

    # If we at least have SQL and some answer text, return a dict
    if sql_code and answer:
        return {
            "sql_code": sql_code,
            "answer": answer,
            "result": result_value,
        }

    return None


def _save_answer_json(base_dir: str, answer: dict) -> None:
    """
    Persist the structured SQL answer to skills/answer-formatting/answer.json
    for easier inspection/debugging.
    """
    try:
        answer_path = os.path.join(
            base_dir, "skills", "answer-formatting", "answer.json"
        )
        os.makedirs(os.path.dirname(answer_path), exist_ok=True)
        with open(answer_path, "w", encoding="utf-8") as f:
            json.dump(answer, f, ensure_ascii=False)
    except Exception as e:  # noqa: PERF203
        print(f"Warning: Could not save answer.json: {e}")


def _extract_structured_answer(result: dict) -> dict | None:
    """
    Scan all Deep Agent messages from the end and find the first one that
    contains a JSON object with sql_code, answer, and result.

    If no such JSON object is found, apply a best-effort heuristic parser
    to markdown-style answers that include:
    - a ```sql ... ``` code block, and
    - a human-readable answer and/or result section.
    """
    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        # First, try strict JSON
        if isinstance(content, str):
            try:
                obj = json.loads(content)
            except Exception:
                obj = None
        elif isinstance(content, dict):
            obj = content
        else:
            obj = None

        if isinstance(obj, dict) and {
            "sql_code",
            "answer",
            "result",
        }.issubset(obj.keys()):
            return obj

        # If not JSON, try to heuristically parse markdown-style content
        if isinstance(content, str):
            heuristic = _parse_markdown_answer(content)
            if heuristic is not None:
                return heuristic

    return None


def get_sql_tool_response(question: str):

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Get or create the cached Deep Agent (with all schema tools)
    agent = _get_sql_agent(base_dir)

    # Keep the user prompt simple. All detailed behavior (how to write SQL,
    # execute it, and format the final structured answer) is defined in skills.
    prompt = (
        "You are a SQL benchmark assistant.\n\n"
        f"User question: {question}\n\n"
    )

    # Retry DeepAgent invocation a few times in case of transient errors
    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )

            # Try to extract structured answer from any message (scanning from the end)
            parsed = _extract_structured_answer(result)
            if parsed is not None:
                result_dict = {
                    "sql_code": parsed.get("sql_code", ""),
                    "answer": parsed.get("answer", ""),
                    "result": parsed.get("result"),
                }
                _save_answer_json(base_dir, result_dict)
                return result_dict

            # If we didn't find structured JSON, fall back to last message content
            messages = result.get("messages") or []
            final_message = messages[-1] if messages else None
            raw_content = (
                getattr(final_message, "content", None)
                if final_message is not None
                else None
            )

            # If not in messages, try to read the answer from the filesystem where
            # the answer-formatting skill may have saved it.
            answer_path = os.path.join(
                base_dir, "skills", "answer-formatting", "answer.json"
            )
            if os.path.exists(answer_path):
                try:
                    with open(answer_path, "r", encoding="utf-8") as f:
                        file_parsed = json.load(f)
                    if (
                        isinstance(file_parsed, dict)
                        and "sql_code" in file_parsed
                        and "answer" in file_parsed
                        and "result" in file_parsed
                    ):
                        return {
                            "sql_code": file_parsed.get("sql_code", ""),
                            "answer": file_parsed.get("answer", ""),
                            "result": file_parsed.get("result"),
                        }
                except Exception as file_err:  # noqa: PERF203
                    print(
                        f"Warning: Failed to read answer-formatting output file: {file_err}"
                    )

            # Fallback: treat whatever we have as a plain-text answer
            if raw_content is not None:
                return {
                    "sql_code": "",
                    "answer": raw_content,
                    "result": None,
                }
        except Exception as e:  # noqa: PERF203
            print(
                f"Error in get_sql_tool_response (attempt {attempt}/{max_retries}): {e}"
            )
            last_error = e

    # All retries failed – return a fallback response with the last error message
    return {
        "sql_code": "",
        "answer": f"Deep agent failed after {max_retries} attempts: {last_error}",
        "result": None,
    }
