"""
Example 1 — Ollama on the same machine as this script (localhost).

Prerequisites: ``pip install -e .`` from repo root, ``ollama pull <LLM_MODEL_NAME>``.

Run from repo root::

    python examples/01_ollama_local.py
"""

from __future__ import annotations

import json
import logging

# ----- Required -------------------------------------------------------------
LLM_MODEL_NAME: str = "deepseek-r1:7b"
EMBEDDING_MODEL: str = "nomic"

JOB_TITLE: str = "Room attendant"
JOB_DESCRIPTION: str = """Cleaning rooms and common areas"""
JOB_QUALIFICATIONS: str = "1+ year hotel experience"

# ----- Optional -------------------------------------------------------------
OLLAMA_HOST: str | None = None  # None = http://127.0.0.1:11434 (or set OLLAMA_HOST env)
VERBOSE_LOGGING: bool = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from esco_skill_extractor import ESCOOccupationMatcher, ESCOSkillExtractor, JobPosting

    matcher = ESCOOccupationMatcher(
        llm_provider="ollama",
        llm_model_name=LLM_MODEL_NAME,
        ollama_host=OLLAMA_HOST,
        embedding_model=EMBEDDING_MODEL,
        verbose_logging=VERBOSE_LOGGING,
    )
    matches = matcher.find_best_occupation(
        job_title=JOB_TITLE,
        description=JOB_DESCRIPTION,
        qualifications=JOB_QUALIFICATIONS,
    )
    matcher.print_results(matches, JOB_TITLE)
    top = matches[0]["occupation_name"] if matches else JOB_TITLE

    extractor = ESCOSkillExtractor(
        llm_provider="ollama",
        llm_model_name=LLM_MODEL_NAME,
        ollama_host=OLLAMA_HOST,
        embedding_model=EMBEDDING_MODEL,
        verbose_logging=VERBOSE_LOGGING,
    )
    skills = extractor.extract_skills(
        JobPosting(
            title=JOB_TITLE,
            esco_occupation_name=top,
            description=JOB_DESCRIPTION,
            qualifications=JOB_QUALIFICATIONS,
        )
    )
    extractor.print_results(skills, JOB_TITLE)
    print("JSON sample:", json.dumps([s.to_dict() for s in skills[:5]], indent=2))


if __name__ == "__main__":
    main()
