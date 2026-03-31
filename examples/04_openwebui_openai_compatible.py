"""
Example 4 — Ollama (or other backends) behind **Open WebUI**, using its **OpenAI-compatible API**.

In Open WebUI: create an API key (Settings → Account / API keys). Use the base URL that serves
the OpenAI-compatible routes (often ``https://<your-host>/api/v1`` or per your deployment docs).

This script uses ``llm_provider="openai"`` with ``openai_base_url`` + your Open WebUI key so the
``openai`` Python client talks to Open WebUI, which forwards to Ollama.

Prerequisites: ``pip install -e .``

Run::

    python examples/04_openwebui_openai_compatible.py
"""

from __future__ import annotations

import json
import logging
import sys

# ----- Required -------------------------------------------------------------
# Open WebUI API key (from Open WebUI UI, not your OpenAI.com key)
OPENWEBUI_API_KEY: str = "sk-YOUR_OPENWEBUI_KEY"

# OpenAI-compatible base URL (must include /v1 if your server expects it)
OPENWEBUI_BASE_URL: str = "https://YOUR_HOST/api/v1"

# Model name as exposed by Open WebUI (often same as Ollama tag, e.g. llama3.2, deepseek-r1:7b)
LLM_MODEL_NAME: str = "gpt-oss:120b"

EMBEDDING_MODEL: str = "nomic"

JOB_TITLE: str = "Room attendant"
JOB_DESCRIPTION: str = "Hotel housekeeping; clean rooms; restock supplies."
JOB_QUALIFICATIONS: str = "Experience in hotels; attention to detail."

# ----- Optional -------------------------------------------------------------
VERBOSE_LOGGING: bool = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from esco_skill_extractor import ESCOOccupationMatcher, ESCOSkillExtractor, JobPosting

    matcher = ESCOOccupationMatcher(
        llm_provider="openai",
        llm_model_name=LLM_MODEL_NAME,
        openai_api_key=OPENWEBUI_API_KEY,
        openai_base_url=OPENWEBUI_BASE_URL,
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
        llm_provider="openai",
        llm_model_name=LLM_MODEL_NAME,
        openai_api_key=OPENWEBUI_API_KEY,
        openai_base_url=OPENWEBUI_BASE_URL,
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
    if OPENWEBUI_API_KEY.startswith("sk-YOUR_") or "YOUR_HOST" in OPENWEBUI_BASE_URL:
        sys.exit(
            "Edit OPENWEBUI_API_KEY and OPENWEBUI_BASE_URL in "
            "examples/04_openwebui_openai_compatible.py"
        )
    main()
