# Structured Data Pipeline

The structured data pipeline extends NeMo Retriever beyond unstructured documents (PDFs, HTML, audio) to **relational databases**. It ingests database schemas from DuckDB into a Neo4j knowledge graph, makes table metadata embeddable and searchable alongside document content, and provides a natural-language text-to-SQL agent.

## Architecture

```
structured_data/               relational_db/
┌──────────────┐    ┌────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│ setup_spider2 │───▶│  DuckDBEngine   │───▶│   extract_data     │───▶│  populate_data      │
│ (one-time)   │    │  (SQL engine)   │    │  (schema → DFs)    │    │  (DFs → Neo4j)      │
└──────────────┘    └────────────────┘    └────────────────────┘    └────────────────────┘
                                                                            │
                           ┌───────────────────────────┐                    │
                           │  prepare_for_embedding     │◀───────────────────┘
                           │  (Neo4j → embedding DFs)   │
                           └─────────────┬─────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │  text_embed pipeline  │  (shared with unstructured docs)
                              └─────────────────────┘

                           ┌───────────────────────────┐
                           │  sql_tool/generate_sql     │  (standalone text-to-SQL agent)
                           │  Deep Agent + DuckDB       │
                           └───────────────────────────┘
```

## Prerequisites

Everything from the main retriever [README](README.md) prerequisites, plus:

- **Docker** (for Neo4j)
- **Git** (for cloning Spider2, or bring your own DuckDB file)
- **Network access** to an LLM endpoint (for the text-to-SQL agent only)

## Step 1 — Install the retriever

From the repo root:

```bash
cd /path/to/nv-ingest
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./nemo_retriever
```

The `relational_db` and `structured_data` dependencies (`duckdb`, `neo4j`, `sqlalchemy`, `langchain-nvidia-ai-endpoints`, `langchain-community`, `deepagents`) are already declared in `nemo_retriever/pyproject.toml`.

## Step 2 — Start Neo4j

```bash
cp .env.example .env
```

Edit `.env` and set a password (minimum 8 characters):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme123
```

Start the Neo4j container:

```bash
docker compose -f nemo_retriever/docker-compose.neo4j.yaml up -d
```

Verify Neo4j is running at http://localhost:7474 (default credentials: `neo4j` / your password).

## Step 3 — Load data into DuckDB

### Option A: Use Spider2 benchmark data (recommended for testing)

The one-time setup script clones the Spider2 repository and loads the spider2-lite databases into a local DuckDB file:

```bash
python nemo_retriever/src/nemo_retriever/structured_data/setup_spider2.py
```

This creates `spider2.duckdb` in the current directory. Optional flags:

```bash
python nemo_retriever/src/nemo_retriever/structured_data/setup_spider2.py \
    --spider2-dir ~/my_spider2 \
    --db ./my.duckdb \
    --overwrite
```

### Option B: Bring your own DuckDB file

If you already have a `.duckdb` file with your schemas and tables, skip the Spider2 setup and point to it in later steps via the `--db` flag or `DUCKDB_PATH` env var.

## Step 4 — Populate Neo4j with schema metadata

Extract tables, columns, primary keys, and foreign keys from DuckDB and write them into Neo4j as a graph (Db → Schema → Table → Column):

```python
from nemo_retriever.relational_db.extract_data import extract_relational_db

extract_relational_db()
```

By default this reads from `./spider2.duckdb`. To use a different database path:

```python
from nemo_retriever.relational_db.extract_data import extract_relational_db
from types import SimpleNamespace

params = SimpleNamespace(db_connection_string="./my.duckdb")
extract_relational_db(params=params)
```

After this step, open the Neo4j browser at http://localhost:7474 and run `MATCH (n) RETURN n LIMIT 50` to verify the graph was populated.

## Step 5 — Generate embeddings from Neo4j table metadata

Once Neo4j is populated, fetch the table metadata and convert it into the same DataFrame format used by the unstructured embedding pipeline:

```python
from nemo_retriever.relational_db.prepare_for_embedding.prepare_embedding_text import (
    fetch_relational_db_for_embedding,
    neo4j_tables_result_to_embedding_dataframe,
)

docs = fetch_relational_db_for_embedding()
df = neo4j_tables_result_to_embedding_dataframe(docs)
print(df.head())
```

Each row contains a `text` field (table name, schema, columns, types) and `metadata` with the Neo4j node ID, making it compatible with the retriever's `text_embed` stage and LanceDB upload.

## Step 6 — Text-to-SQL (optional)

The SQL tool lets you ask natural language questions against the DuckDB data:

```bash
export DUCKDB_PATH=./spider2.duckdb
export LLM_INVOKE_URL=https://integrate.api.nvidia.com/v1   # or your endpoint
export LLM_API_KEY=nvapi-...
export LLM_MODEL=meta/llama-3.1-70b-instruct               # optional, this is the default
```

```python
from nemo_retriever.relational_db.sql_tool.generate_sql import get_sql_tool_response

result = get_sql_tool_response("How many flights are in the Airlines database?")
print(result["sql_code"])   # SELECT COUNT(*) FROM Airlines.flights
print(result["answer"])     # There are 1234 flights...
print(result["result"])     # 1234
```

The agent discovers all schemas in the DuckDB file, creates per-schema SQL toolkits, and uses a LangChain Deep Agent to plan and execute queries.

## Environment variables reference

| Variable | Default | Used by |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USERNAME` | `neo4j` | Neo4j connection |
| `NEO4J_PASSWORD` | *(required)* | Neo4j connection |
| `DUCKDB_PATH` | `./spider2.duckdb` | `sql_tool/generate_sql.py` |
| `LLM_INVOKE_URL` | *(required for SQL tool)* | ChatNVIDIA endpoint |
| `LLM_API_KEY` | *(required for SQL tool)* | ChatNVIDIA auth |
| `LLM_MODEL` | `meta/llama-3.1-70b-instruct` | LLM model name |

## Package layout

```
nemo_retriever/src/nemo_retriever/
├── structured_data/
│   ├── duckdb_engine.py          # DuckDBEngine: in-process SQL + catalog introspection
│   ├── spider2_loader.py         # Load Spider2-lite JSON into DuckDB schemas
│   └── setup_spider2.py          # One-time CLI: clone Spider2 + load into DuckDB
│
├── relational_db/
│   ├── extract_data.py           # Bridge: DuckDB → DataFrames → populate Neo4j
│   ├── neo4j_connection/
│   │   └── store.py              # Neo4jConnection, Neo4jConnectionManager, get_neo4j_conn()
│   ├── population/
│   │   ├── populate_data.py      # Top-level: populate_structured_data()
│   │   ├── db/dal.py             # Diff-based schema sync (add/update/delete)
│   │   └── graph/
│   │       ├── model/            # Node, Schema, Labels
│   │       ├── parsers/          # DataFrames → Schema objects
│   │       ├── dal/              # Neo4j CRUD (merge nodes, edges, FKs, PKs)
│   │       ├── services/         # add_schema() high-level service
│   │       ├── indexes.py        # Neo4j constraints and full-text indexes
│   │       └── utils.py          # DataFrame normalization helpers
│   ├── prepare_for_embedding/
│   │   └── prepare_embedding_text.py  # Neo4j → embedding-ready DataFrame
│   └── sql_tool/
│       └── generate_sql.py       # Text-to-SQL via Deep Agent + DuckDB
```

## Troubleshooting

**Neo4j won't start** — Check Docker logs: `docker compose -f nemo_retriever/docker-compose.neo4j.yaml logs`. Ensure ports 7474 and 7687 are free and the password in `.env` is at least 8 characters.

**`ImportError: No module named 'duckdb'`** — Run `uv pip install duckdb>=1.2.0` inside your `.retriever` venv.

**`extract_relational_db()` fails with connection errors** — Verify Neo4j is running and `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` are set (either in `.env` or as environment variables).

**`setup_spider2.py` fails to clone** — Ensure you have git installed and network access to GitHub. If behind a proxy, set `https_proxy` before running.

**SQL tool returns empty answers** — Verify `DUCKDB_PATH` points to a populated DuckDB file, and `LLM_INVOKE_URL` / `LLM_API_KEY` are set to a reachable LLM endpoint.
