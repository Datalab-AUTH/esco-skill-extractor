"""Paths to ESCO CSV files shipped inside the package."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def bundled_esco_csv(filename: str) -> str:
    """
    Absolute path to a CSV file under ``esco_skill_extractor/data/``.

    Common names: ``esco_skills.csv``, ``esco_occupations.csv``,
    ``occupation_skills_mapping.csv``.
    """
    path = package_root() / "data" / filename
    return str(path.resolve())
