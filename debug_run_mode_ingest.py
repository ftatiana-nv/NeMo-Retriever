#!/usr/bin/env python3
"""Local debug script to run run_mode_ingest. Run from repo root with PYTHONPATH=nemo_retriever/src.

To see embedding models allowed for your NVIDIA API key: python debug_run_mode_ingest.py --list-models

To debug .embed and .vdb_upload on the structured path:
- embed_params and vdb_params below are required; otherwise ingest_structured() runs
  fetch_structured() but has no tasks and skips embed + vdb_upload.
- Exact call sites and I/O: see EMBED_VDB_DEBUG_GUIDE.md.
- Breakpoints: inprocess.run_pipeline_tasks_on_df before/after the line that calls
  embed_text_main_text_embed (input/output DataFrame); inprocess._embed_group line with
  model.embed(batch, ...) for the actual embed input (list of strings).
- To compare with PDF path: run
  PYTHONPATH=nemo_retriever/src python -m nemo_retriever.examples.inprocess_pipeline doc /path/to/file.pdf
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nemo_retriever.application.modes.executor import run_mode_ingest
from nemo_retriever.params import (
    EmbedParams,
    IngestExecuteParams,
    IngestorCreateParams,
    StructuredExtractParams,
    VdbUploadParams,
)


def _check_embed_key() -> None:
    """Print whether embedding API key is set (masked). Run before ingest to debug 401."""
    key = os.environ.get("NVIDIA_API_KEY") or ""
    if not key or not key.strip():
        print("DEBUG: No embedding API key found. Set LLM_API_KEY or NVIDIA_API_KEY (or load .env).")
        return
    # Mask: show only first 4 and last 4 chars
    k = key.strip()
    if len(k) <= 12:
        masked = "*" * len(k)
    else:
        masked = f"{k[:4]}...{k[-4:]}"
    print(f"DEBUG: Embedding API key is set ({masked}, len={len(k)}). If you still get 401, the key may be invalid or lack access to the embedding model.")


def list_embedding_models() -> None:
    """Call NVIDIA Inference API GET /v1/models and print embedding models for your key."""
    base = os.environ.get("NVIDIA_INFERENCE_BASE", "https://inference-api.nvidia.com")
    key = ( os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not key:
        print("Set LLM_API_KEY or NVIDIA_API_KEY (e.g. in .env) then run again.")
        return
    url = f"{base.rstrip('/')}/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"GET {url} failed: {e.code} {e.reason}")
        if e.fp:
            print(e.fp.read().decode())
        return
    models = data.get("data") if isinstance(data, dict) else []
    if not models:
        print("No models in response. Raw:", data)
        return
    # Prefer models whose id contains 'embed'; show all if none match.
    ids = [m.get("id") for m in models if m.get("id")]
    embed_ids = [i for i in ids if "embed" in i.lower()]
    show = embed_ids if embed_ids else ids
    print("Available models for your key (embedding-related or all):")
    for i in sorted(show):
        print(f"  {i}")
    if embed_ids:
        print("\nSet embed_params.model_name to one of the ids above (e.g. the first).")
    else:
        print("\nNo model id contains 'embed'; check NVIDIA docs for which of the above support embeddings.")


def main() -> None:
    if "--list-models" in sys.argv:
        list_embedding_models()
        return
    _check_embed_key()  # remove after debugging 401

    # Inprocess mode. Pass txt file(s) to run the txt pipeline (extract_txt -> embed -> vdb_upload).
    create_params = IngestorCreateParams(
        documents=[],  # or multiple: ["./data/a.txt", "./data/b.txt"]  "/Users/tfrenklach/Desktop/NeMo-Retriever/data/test.txt"
    )
    ingest_params = IngestExecuteParams(show_progress=True)

    # Optional: run structured extract (DuckDB + Neo4j). Requires Neo4j at NEO4J_URI (default localhost:7687).
    # Set to None to skip structured path when Neo4j is not running.
    _structured_params_ready = StructuredExtractParams(db_connection_string="./spider2.duckdb")
    structured_params = _structured_params_ready 

    # Required for structured path to run embed + vdb_upload. Without these,
    # ingest_structured() only runs fetch_structured() and returns the DataFrame.
    # NVIDIA Inference API. Use exact id from --list-models (e.g. nvidia/nvidia/... for inference-api.nvidia.com).
    embed_params = EmbedParams(
        model_name="nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2",
        embed_modality="text",
        text_column="text",
        inference_batch_size=16,
        embedding_endpoint="https://inference-api.nvidia.com",
        embedding_api_key=os.environ.get("NVIDIA_API_KEY"),
    )
    # Unstructured path (documents → extract_txt → embed → vdb_upload) writes here.
    vdb_params = VdbUploadParams(
        lancedb=dict(
            lancedb_uri="lancedb",
            table_name="nv-ingest",
            overwrite=True,
            create_index=True,
        )
    )

    result, structured_future = run_mode_ingest(
        run_mode="inprocess",
        create_params=create_params,
        ingest_params=ingest_params,
        structured_params=structured_params,
        embed_params=embed_params,
        vdb_params=vdb_params,
    )

    print("ingest() result:", result)

    if structured_future is not None:
        print("Waiting for structured ingest to finish...")
        try:
            structured_future.result(timeout=300)
            print("Structured ingest completed.")
        except Exception as e:  # noqa: BLE001
            print("Structured ingest failed:", e)
            if structured_future.exception() is not None:
                raise structured_future.exception() from e


if __name__ == "__main__":
    main()
