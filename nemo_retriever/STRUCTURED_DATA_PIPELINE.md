# Structured Data Pipeline

This document covers the structured-data work added in the `staging` branch.
Two independent capabilities were added:

- **Path A — Schema ingestion:** reads a DuckDB database, writes schema metadata
  into a Neo4j schema graph, and converts that graph data into embeddings so it
  can be searched alongside document content in LanceDB.
- **Path B — text-to-SQL agent:** queries DuckDB directly with natural language
  via a Deep Agent. Does not depend on Neo4j.

```
Path A — Schema ingestion

  spider2.duckdb
      │
      ▼
  extract_relational_db()          ← orchestrates extraction AND population:
      │                               reads tables/columns/PKs/FKs from DuckDB,
      │                               then writes them into Neo4j
      ▼
  Neo4j schema graph
  (Db)-[:CONTAINS]->(Schema)
       -[:CONTAINS]->(Table)
       -[:CONTAINS]->(Column)
      │
      ▼
  fetch_relational_db_for_embedding()   ← Cypher query → list of dicts
      │
      ▼
  neo4j_tables_result_to_embedding_dataframe()
      │
      ▼
  embed()  →  vdb_upload()  →  LanceDB table "nv-ingest-structured"


Path B — text-to-SQL  (independent of Neo4j)

  spider2.duckdb  +  LLM endpoint
      │
      ▼
  get_sql_tool_response("your question")
      │
      ▼
  { sql_code, answer, result }
```

---

## Pipeline step status

`InProcessIngestor.ingest_structured()` defines an 8-step pipeline.
Current implementation status:

| Step | Method | Status |
|------|--------|--------|
| 1 | `extract_structured` | ✅ Implemented — DuckDB schema → Neo4j graph |
| 2 | `populate_structured_semantic_layer` | ⚠️ Stub |
| 3 | `detect_structured_pii` | ⚠️ Stub |
| 4 | `populate_structured_usage_weights` | ⚠️ Stub |
| 5 | `generate_structured_descriptions` | ⚠️ Stub |
| 6 | `fetch_structured` | ✅ Implemented — Neo4j → embedding-ready DataFrame |
| 7 | `embed` | ✅ Implemented — shared with unstructured pipeline |
| 8 | `vdb_upload` | ✅ Implemented — writes to LanceDB table `nv-ingest-structured` |

---

## Known gaps and future work

The following items are either unimplemented stubs or known bugs in the current
staging code. Anyone picking up this work should expect to address these.

### Steps 2–5 — Unimplemented pipeline stubs

All four methods exist in `InProcessIngestor` and are called by `ingest_structured()`
but their bodies are `pass`. They are no-ops today.

| Step | What it is supposed to do when implemented |
|------|---------------------------------------------|
| `populate_structured_semantic_layer` | Match global business terms and attributes to `Table` and `Column` nodes in Neo4j; create `Term`/`Attribute` nodes with `MAPS_TO_TABLE` / `MAPS_TO_COLUMN` relationships for anything unmatched |
| `detect_structured_pii` | Apply regex patterns (and optionally an LLM call) to column names/descriptions; tag matching `Column` nodes with a `pii_type` property and a `HAS_PII_TYPE` relationship |
| `populate_structured_usage_weights` | Parse SQL query log files, compute table/column co-occurrence frequencies, and write `usage_weight` float properties back onto the corresponding graph nodes |
| `generate_structured_descriptions` | Call an LLM to generate natural-language descriptions for every `Db`, `Schema`, `Table`, `Column`, `View`, and `Query` node; write results back to Neo4j as a `description` property |

Each stub already has a corresponding `Params` class in `params/models.py`
(`StructuredSemanticLayerParams`, `StructuredPIIParams`, `StructuredUsageWeightsParams`,
`StructuredDescriptionParams`) that defines the expected configuration interface.

### `db_connection_string` not wired through

`extract_relational_db(params=StructuredExtractParams(db_connection_string="..."))` 
accepts a custom database path and passes it to `settings`, but `create_dataframe()`
ignores `settings` and always opens `./spider2.duckdb` directly.

**Fix needed:** `create_dataframe()` should read the database path from the `settings`
argument it already receives instead of hardcoding the path.

### `main()` not defined in `extract_data.py`

The file ends with:
```python
if __name__ == "__main__":
    main()
```
but no `main()` function is defined anywhere in the file. Running it as a script
raises `NameError: name 'main' is not defined`.

**Fix needed:** either define a `main()` entry point, or remove the `__main__` block.

### FK relationship label unverified in Cypher

The Cypher queries in this document use `[:FK]` for foreign key relationships.
This label was not verified against the current staging code after the PascalCase
node-label rename. If the FK queries return no results, check the actual relationship
type with:

```cypher
MATCH ()-[r]->() RETURN DISTINCT type(r)
```

and update the queries accordingly.

---

## Prerequisites

- Docker (for Neo4j — Path A only)
- Git (for cloning Spider2, or bring your own DuckDB file)
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- LLM endpoint + API key (for the text-to-SQL agent — Path B only)

---

## Step 1 — Install the package

From the repo root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./nemo_retriever
```

> ⚠️ **Namespace package trap.** The repo root contains a `nemo_retriever/`
> directory (Docker configs, compose files). If Python runs from the repo root
> without the editable install, it resolves `import nemo_retriever` to that
> directory instead of the installed package, and `nemo_retriever.relational_db`
> will not be found. Running `uv pip install -e ./nemo_retriever` fixes this.

All dependencies (`duckdb`, `neo4j`, `sqlalchemy`, `langchain-nvidia-ai-endpoints`,
`langchain-community`, `deepagents`) are declared in `nemo_retriever/pyproject.toml`
and installed by the above command.

---

## Step 2 — Configure credentials

> ⚠️ **Do this before starting Docker.** The `docker-compose.neo4j.yaml` reads
> `.env` on container startup to set `NEO4J_AUTH`. Starting Docker first results
> in wrong or missing credentials.

```bash
cp .env.example .env
```

Edit `.env`:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme123
```

> ⚠️ **Avoid spaces in `NEO4J_PASSWORD`.** The command `export $(cat .env | xargs)`
> splits on whitespace and will silently truncate a password like `my pass` to
> `my`. Use a single-word password, or load the file as shown below.

Export the variables into your shell. The Python connection manager reads
`os.environ` directly — having a `.env` file alone is not enough:

```bash
set -a
source .env
set +a
```

---

## Step 3 — Start Neo4j

The graph population code uses APOC procedures (`apoc.merge.node.eager`,
`apoc.merge.relationship`). The provided Compose file enables APOC and Graph
Data Science automatically.

```bash
docker compose -f nemo_retriever/docker-compose.neo4j.yaml up -d
```

Wait ~30 seconds, then verify the container is healthy:

```bash
docker compose -f nemo_retriever/docker-compose.neo4j.yaml ps
```

| Interface | URL |
|---|---|
| Browser UI | http://localhost:7474 |
| Bolt (Python) | `bolt://localhost:7687` |

---

## Step 4 — Load Spider2 into DuckDB

### Option A — Use Spider2 benchmark data (recommended for testing)

The one-time setup script clones the Spider2 repository and loads the
`spider2-lite` databases into a local DuckDB file:

```bash
python nemo_retriever/src/nemo_retriever/structured_data/setup_spider2.py
```

This creates `spider2.duckdb` in the **current working directory**.
The Spider2 repo is cloned to `~/spider2` by default.

Optional flags:

```bash
python nemo_retriever/src/nemo_retriever/structured_data/setup_spider2.py \
    --spider2-dir ~/my_spider2 \
    --db ./my.duckdb \
    --overwrite          # drop and recreate schemas that already exist
```

### Option B — Bring your own DuckDB file

Place (or symlink) your file as `./spider2.duckdb` in the repo root.

> ⚠️ **Known limitation: `db_connection_string` is not wired through.**
> `extract_relational_db()` accepts `params.db_connection_string` and computes
> a `db_path` from it, but `create_dataframe()` ignores the `settings` argument
> and always opens `./spider2.duckdb` directly. For now, rename or symlink your
> file to `./spider2.duckdb` regardless of what path you pass in params.

---

## Step 5 — Run the pipeline

### Option A — Full pipeline (Neo4j population + embed + LanceDB upload)

```python
from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams, StructuredExtractParams, VdbUploadParams

ingestor = (
    create_ingestor(run_mode="inprocess")
    .embed(EmbedParams(embedding_endpoint="http://localhost:8012/v1"))
    .vdb_upload(VdbUploadParams())
)

ingestor.ingest_structured(StructuredExtractParams())
```

This runs all 8 steps in order. Stubs (2–5) are no-ops. The result is
written to LanceDB table `nv-ingest-structured`.

To embed with a local HuggingFace model instead of a remote endpoint:

```python
EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2")
```

### Option B — Schema extraction and graph population only

`extract_relational_db()` does both steps in one call: it reads tables, columns,
PKs, and FKs from DuckDB, then writes all of them into Neo4j as a schema graph.

```python
from nemo_retriever.relational_db.extract_data import extract_relational_db

extract_relational_db()
```

> ⚠️ `extract_data.py` ends with `if __name__ == "__main__": main()` but
> **no `main()` function is defined**. Run via import only, not as a script.

### Option C — Graph population + inspect embedding DataFrame

Useful for debugging what the pipeline produces before running embed:

```python
from nemo_retriever.relational_db.extract_data import extract_relational_db
from nemo_retriever.relational_db.prepare_for_embedding.prepare_embedding_text import (
    fetch_relational_db_for_embedding,
    neo4j_tables_result_to_embedding_dataframe,
)

extract_relational_db()

docs = fetch_relational_db_for_embedding()
df = neo4j_tables_result_to_embedding_dataframe(docs)
print(df[["text", "path", "metadata"]].head())
```

Each row contains:
- `text` — e.g. `"db_name: spider2, schema_name: Airlines, table_name: flights, columns: ..."`
- `_embed_modality` — `"text"`
- `path` — `"neo4j:<node_id>"`
- `page_number` — `-1`
- `metadata` — `{id, label, name, source_path}`

---

## Step 6 — Verify the Neo4j graph

Open http://localhost:7474, log in with your `.env` credentials, and run:

```cypher
-- Count all nodes by type
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS total ORDER BY total DESC

-- Explore the schema hierarchy
MATCH (d:Db)-[:CONTAINS]->(s:Schema)-[:CONTAINS]->(t:Table)-[:CONTAINS]->(c:Column)
RETURN d.name, s.name, t.name, count(c) AS columns
ORDER BY columns DESC
LIMIT 25

-- Check foreign keys
MATCH (c1:Column)-[:FK]->(c2:Column)
RETURN c1.name AS fk_column, c2.name AS pk_column
LIMIT 20
```

Or verify from Python:

```python
from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn

conn = get_neo4j_conn()
conn.verify_connectivity()
result = conn.query_read_only("MATCH (n) RETURN count(n) AS total", {})
print(result)
```

---

## Step 7 — text-to-SQL agent (optional, Path B)

Independent of Neo4j. Connects to DuckDB via SQLAlchemy, discovers all schemas,
and uses a LangChain Deep Agent to answer natural language questions with SQL.

Set the required environment variables:

```bash
export DUCKDB_PATH=./spider2.duckdb
export LLM_INVOKE_URL=https://integrate.api.nvidia.com/v1   # or your endpoint
export LLM_API_KEY=nvapi-...
export LLM_MODEL=meta/llama-3.1-70b-instruct                # optional, this is the default
```

Run a query:

```python
from nemo_retriever.relational_db.sql_tool.generate_sql import get_sql_tool_response

result = get_sql_tool_response("How many flights are in the Airlines database?")
print(result["sql_code"])    # Generated SQL
print(result["answer"])      # Natural language answer
print(result["result"])      # Extracted result value
```

The agent discovers all schemas in the DuckDB file on first call, builds one
`SQLDatabaseToolkit` per schema (cached for the process lifetime), and retries
the Deep Agent invocation up to 3 times on transient errors.

---

## Environment variables reference

| Variable | Default | Required for |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Path A — schema ingestion |
| `NEO4J_USERNAME` | `neo4j` | Path A — schema ingestion |
| `NEO4J_PASSWORD` | *(required)* | Path A — schema ingestion |
| `DUCKDB_PATH` | `./spider2.duckdb` | Path B — text-to-SQL agent |
| `LLM_INVOKE_URL` | *(required for Path B)* | text-to-SQL agent |
| `LLM_API_KEY` | *(required for Path B)* | text-to-SQL agent |
| `LLM_MODEL` | `meta/llama-3.1-70b-instruct` | text-to-SQL agent |

---

## Package layout

```
nemo_retriever/src/nemo_retriever/
│
├── structured_data/
│   ├── duckdb_engine.py          # DuckDBEngine: in-process SQL + catalog introspection
│   ├── spider2_loader.py         # Load Spider2-lite JSON → DuckDB schemas
│   └── setup_spider2.py          # One-time CLI: clone Spider2 + load into DuckDB
│
├── relational_db/
│   ├── extract_data.py           # Entry point: DuckDB introspection → Neo4j population
│   ├── neo4j_connection/
│   │   └── store.py              # Neo4jConnection, Neo4jConnectionManager, get_neo4j_conn()
│   ├── population/
│   │   ├── populate_data.py      # Top-level orchestrator: populate_structured_data()
│   │   ├── db/dal.py             # Diff-based schema sync (add/update/delete nodes)
│   │   └── graph/
│   │       ├── model/            # Node, Schema, Labels (Db/Schema/Table/Column)
│   │       ├── parsers/          # DataFrames → Schema objects
│   │       ├── dal/              # Neo4j CRUD (merge nodes, edges, FKs, PKs)
│   │       ├── services/         # add_schema() high-level orchestration
│   │       ├── indexes.py        # Neo4j constraints and fulltext indexes
│   │       └── utils.py          # DataFrame normalization helpers
│   ├── prepare_for_embedding/
│   │   └── prepare_embedding_text.py  # Neo4j table metadata → embedding-ready DataFrame
│   └── sql_tool/
│       └── generate_sql.py       # Text-to-SQL via Deep Agent + SQLAlchemy + DuckDB
│
├── ingest_modes/
│   └── inprocess.py              # InProcessIngestor.ingest_structured() — 8-step pipeline
│
└── params/
    └── models.py                 # StructuredExtractParams, StructuredFetchParams, etc.
```

---

## Day-to-day Docker commands

```bash
# Start Neo4j (data preserved in named volume)
docker compose -f nemo_retriever/docker-compose.neo4j.yaml up -d

# Stop Neo4j
docker compose -f nemo_retriever/docker-compose.neo4j.yaml down

# Wipe all graph data and start fresh
docker compose -f nemo_retriever/docker-compose.neo4j.yaml down -v
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'nemo_retriever.relational_db'`**
Python is resolving `nemo_retriever` to the `nemo_retriever/` directory at the repo
root instead of the installed package. Run `uv pip install -e ./nemo_retriever` and
make sure you are using the venv's Python binary.

**`KeyError: 'NEO4J_URI'`**
The Python code reads credentials from `os.environ` directly. Having a `.env` file
is not enough. Export them first: `set -a && source .env && set +a`

**Credentials silently wrong after `export $(cat .env | xargs)`**
If your password contains spaces, `xargs` splits on them and truncates the value.
Use `set -a && source .env && set +a` instead.

**`extract_relational_db()` writes nothing to Neo4j**
Confirm `./spider2.duckdb` exists in your **current working directory**. The path
is hardcoded in `create_dataframe()` regardless of what you pass to `db_connection_string`.

**`ClientError: Unknown procedure apoc.merge.node.eager`**
You are not using the provided `docker-compose.neo4j.yaml`. APOC must be enabled.
Start Neo4j with: `docker compose -f nemo_retriever/docker-compose.neo4j.yaml up -d`

**`extract_data.py: main() not defined`**
Do not run `extract_data.py` as a script. Call `extract_relational_db()` via import.

**Neo4j container unhealthy**
Check logs: `docker compose -f nemo_retriever/docker-compose.neo4j.yaml logs neo4j`
Allow up to 60 seconds on first run. Ensure ports 7474 and 7687 are free.

**Password mismatch after changing `.env`**
Recreate the container to re-apply the password:
```bash
docker compose -f nemo_retriever/docker-compose.neo4j.yaml down -v
docker compose -f nemo_retriever/docker-compose.neo4j.yaml up -d
```

**`setup_spider2.py` fails to clone**
Ensure git is installed and you have network access to GitHub.
If behind a proxy: `export https_proxy=http://your-proxy:port`

**SQL tool returns empty answers**
Verify `DUCKDB_PATH` points to a populated `.duckdb` file and that
`LLM_INVOKE_URL` / `LLM_API_KEY` are set and reachable.
