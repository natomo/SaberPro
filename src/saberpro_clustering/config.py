"""Carga y validación de la configuración del pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Carga el archivo de configuración YAML del proyecto.

    Args:
        path: Ruta al archivo config.yaml.

    Returns:
        Diccionario con la configuración completa.

    Raises:
        FileNotFoundError: si el archivo de configuración no existe.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración en '{config_path}'. "
            "Copia config.yaml.example o ajusta la ruta."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
