# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent


def load_custom_analyses(path: Path | str | None = None) -> list[dict]:
    """Load custom analyses (gold SQL snippets) from YAML.

    Returns a list of dicts, each with keys: name, description, sql,
    and optionally tables and tags.
    """
    path = Path(path) if path else _CONFIG_DIR / "custom_analyses.yaml"
    if not path.exists():
        logger.debug("Custom analyses file not found: %s", path)
        return []

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    analyses = data.get("custom_analyses", [])
    if not analyses:
        logger.debug("No custom_analyses entries found in %s", path)
    else:
        logger.info("Loaded %d custom analyses from %s", len(analyses), path)
    return analyses
