"""Repository paths used by CLI defaults and model adapters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def weights_path(name: str) -> str:
    """Resolve bare model file names inside the repository weights directory."""
    path = Path(name)
    return str(WEIGHTS_DIR / name) if path.parent == Path(".") else name
