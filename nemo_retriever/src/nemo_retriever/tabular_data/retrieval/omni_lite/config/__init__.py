import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent


def load_acronyms(path: Path | str | None = None) -> list[dict]:
    """Load acronyms from YAML. Returns list of {name, description} dicts
    compatible with the ontology 'dictionary' format."""
    path = Path(path) if path else _CONFIG_DIR / "acronyms.yaml"
    if not path.exists():
        logger.warning("Acronyms file not found: %s", path)
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("acronyms", [])
    result = []
    for entry in raw:
        acronym = entry.get("acronym", "")
        definition = entry.get("definition", "")
        data_objects = entry.get("data_objects", [])
        desc = f"{acronym} — {definition}"
        if data_objects:
            desc += f". Related data objects: {', '.join(data_objects)}"
        result.append({"name": acronym, "description": desc})
    return result


def load_custom_prompts(path: Path | str | None = None) -> str:
    """Load custom prompt add-ons from YAML. Returns a single string
    with all prompts concatenated, ready to prepend to user queries."""
    path = Path(path) if path else _CONFIG_DIR / "custom_prompts.yaml"
    if not path.exists():
        logger.warning("Custom prompts file not found: %s", path)
        return ""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("custom_prompts", [])
    prompts = [entry.get("prompt", "") for entry in raw if entry.get("prompt")]
    return "\n".join(prompts)


def load_ontology(
    acronyms_path: Path | str | None = None,
    custom_prompts_path: Path | str | None = None,
    industry: list[str] | None = None,
) -> dict:
    """Build ontology dict from config files.

    Args:
        acronyms_path: path to acronyms YAML (default: config/acronyms.yaml)
        custom_prompts_path: path to custom prompts YAML (default: config/custom_prompts.yaml)
        industry: list of industry names for context
    """
    acronyms = load_acronyms(acronyms_path)
    custom_prompts = load_custom_prompts(custom_prompts_path)
    return {
        "industry": industry or [],
        "dictionary": acronyms,
        "overview": custom_prompts,
    }
