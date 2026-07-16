from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def project_directories() -> None:
    ensure_directories([RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR])


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def processed_path(*parts: str) -> Path:
    return PROCESSED_DIR.joinpath(*parts)


def figures_path(*parts: str) -> Path:
    return FIGURES_DIR.joinpath(*parts)


def models_path(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)


def save_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_env_file(env_path: Path | None = None) -> dict[str, str]:
    source = env_path or PROJECT_ROOT / ".env.local"
    if not source.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values
