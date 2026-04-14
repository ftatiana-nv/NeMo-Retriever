import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent


def load_acronyms(path: Path | str | None = None) -> str:
    """Load acronyms from YAML. Returns a newline-joined string of '{acronym} - {definition}'."""
    path = Path(path) if path else _CONFIG_DIR / "acronyms.yaml"
    if not path.exists():
        return ""

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("acronyms", [])
    result = []
    for entry in raw:
        text = f"{entry.get('acronym', '')} - {entry.get('definition', '')}"
        data_objects = entry.get("data_objects")
        if data_objects:
            text += f". Related data objects: {', '.join(data_objects)}"
        result.append(text)
    return "\n".join(result)


def load_custom_prompts(path: Path | str | None = None) -> str:
    """Load custom prompt add-ons from YAML. Returns a newline-joined string of prompts."""
    path = Path(path) if path else _CONFIG_DIR / "custom_prompts.yaml"
    if not path.exists():
        return ""

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("custom_prompts", [])
    prompts = [entry.get("prompt", "") for entry in raw if entry.get("prompt")]
    return "\n".join(prompts)
