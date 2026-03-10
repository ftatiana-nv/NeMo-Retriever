# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import RunMode
from nemo_retriever.params import StructuredExtractParams

from .factory import create_runmode_ingestor

# Module-level executor so the background thread outlives the caller's stack
# frame and is not killed when ingest() returns.
_structured_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="structured-ingest")


def run_mode_ingest(
    *,
    run_mode: RunMode,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
    structured_params: StructuredExtractParams | None = None,
) -> tuple[object, Optional[Future]]:
    ingestor = create_runmode_ingestor(run_mode=run_mode, params=create_params)

    structured_future: Optional[Future] = None
    if structured_params is not None:
        # Submit to the module-level executor so the thread survives past this
        # function's return and is not killed if the main thread finishes first.
        # Callers can inspect structured_future.result() / .exception() later.
        structured_future = _structured_executor.submit(
            ingestor.ingest_structured, structured_params
        )

    return ingestor.ingest(params=ingest_params), structured_future
