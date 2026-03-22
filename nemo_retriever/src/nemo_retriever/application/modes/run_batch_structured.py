# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.params import EmbedParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import StructuredExtractParams
from nemo_retriever.params import VdbUploadParams

from .executor import run_mode_ingest_structured


def run_batch_structured(
    *,
    create_params: IngestorCreateParams | None = None,
    structured_params: StructuredExtractParams,
    embed_params: EmbedParams | None = None,
    vdb_params: VdbUploadParams | None = None,
) -> object:
    return run_mode_ingest_structured(
        run_mode="batch",
        create_params=create_params,
        structured_params=structured_params,
        embed_params=embed_params,
        vdb_params=vdb_params,
    )
