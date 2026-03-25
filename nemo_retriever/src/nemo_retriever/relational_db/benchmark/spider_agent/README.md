# Spider-Agent (upstream) benchmark hook

This folder does **not** vendor [Spider-Agent-Lite](https://github.com/xlang-ai/Spider2/tree/main/methods/spider-agent-lite). It:

1. Requires a local clone of [`xlang-ai/Spider2`](https://github.com/xlang-ai/Spider2).
2. Sets **`SPIDER2_REPO_ROOT`** to that clone’s root.
3. Runs `methods/spider-agent-lite/run.py` (Docker + cloud credentials as in their README).
4. Harvests `generated_sql/spider_agent/<instance_id>.sql` from `output/.../spider/result.json`.

## Python: install the Docker **SDK** (not the same as Docker Desktop)

Spider-Agent-Lite does `import docker` (PyPI package **`docker`**). Your NeMo `.venv` must have it:

```bash
cd /path/to/NeMo-Retriever
.venv/bin/python -m pip install "docker>=7"
# or: uv pip install -e "nemo_retriever[spider-agent]"   # if you use uv from repo layout
```

Keep **Docker Desktop** (or `dockerd`) running; the Python package talks to that daemon.

## Python: install the **`openai`** package (not the same as “using OpenAI”)

Upstream imports `AzureOpenAI` / `OpenAI` from the **`openai`** library on PyPI. Install it in the **same venv** as NeMo:

```bash
.venv/bin/python -m pip install "openai>=1"
```

That does **not** mean you must use OpenAI’s cloud API; see [NVIDIA_OPENAI_COMPAT.md](NVIDIA_OPENAI_COMPAT.md) to use **`LLM_INVOKE_URL`** + **`LLM_API_KEY`** from `.env` (you may need a small patch in your Spider2 clone’s `models.py`).

## Quick start

From the repo root, load your LLM settings from `.env` (same variables as the rest of NeMo-Retriever: **`LLM_MODEL`**, `LLM_API_KEY`, `LLM_INVOKE_URL`, etc.). Upstream `--model` is chosen in this order: **`SPIDER_AGENT_MODEL`** (if set), then **`LLM_MODEL`**, then **`OPENAI_MODEL`**, else `gpt-4o`.

```bash
cd /path/to/NeMo-Retriever
set -a && source .env && set +a          # exports LLM_MODEL and keys from .env
export SPIDER2_REPO_ROOT=$HOME/Spider2   # your Spider2 clone

# Optional: only local SQLite/DuckDB tasks (matches spider2-lite “local*” rows)
export SPIDER_AGENT_LOCAL_ONLY=1

PYTHONPATH=nemo_retriever/src python debug_retriever_sql.py --batch --mode spider-agent --prefix local
```

**VS Code:** use the launch configuration **“retriever: debug SQL batch (spider-agent)”** (loads `${workspaceFolder}/.env`). Add a line to `.env`:

`SPIDER2_REPO_ROOT=/absolute/path/to/Spider2`

To force a different model for Spider-Agent only, set **`SPIDER_AGENT_MODEL`** in `.env` (it overrides `LLM_MODEL` for this pipeline).

Follow upstream [methods/spider-agent-lite/README](https://github.com/xlang-ai/Spider2/blob/main/methods/spider-agent-lite/README.md) for conda deps, Docker, BigQuery/Snowflake JSON credentials, and `spider_agent_setup_lite.py`.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `SPIDER2_REPO_ROOT` | **Required.** Root of Spider2 clone. |
| `SPIDER_AGENT_MODEL` | Optional. Overrides `LLM_MODEL` for upstream `--model`. |
| `LLM_MODEL` | From **`.env`**: primary model id for NeMo-Retriever; used if `SPIDER_AGENT_MODEL` is unset. |
| `OPENAI_MODEL` | Fallback if neither of the above is set (some setups use this name in `.env`). |
| `SPIDER_AGENT_SUFFIX` | Experiment suffix `-s` (default `nemo-benchmark`). |
| `SPIDER_AGENT_OUTPUT_DIR` | Override upstream output dir (default: `<spider-agent-lite>/output`). |
| `SPIDER_AGENT_LOCAL_ONLY` | `1` (default): `--local_only`; `0`: run BQ/SF tasks (needs creds). |
| `SPIDER_AGENT_INSTANCE_ID` | Single-shot debug id (e.g. `local001`). |
