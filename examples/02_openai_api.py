"""
Example 2 — OpenAI cloud API (chat + local ``nomic`` embeddings by default).

Prerequisites: ``pip install -e .``, valid ``OPENAI_API_KEY``.

Run::

    python examples/02_openai_api.py
"""

from __future__ import annotations

import json
import logging
import sys

# ----- Required -------------------------------------------------------------
OPENAI_API_KEY: str = "sk-YOUR_OPENAI_KEY"  # change me

# Chat model on api.openai.com
LLM_MODEL_NAME: str = "gpt-4o-mini"

# Local Sentence-Transformers embeddings (no extra OpenAI spend), or "openai"
EMBEDDING_MODEL: str = "nomic"

JOB_TITLE: str = "Software developer"
JOB_DESCRIPTION: str = "Build APIs in Python; PostgreSQL; code review; agile team."
JOB_QUALIFICATIONS: str = "3+ years Python; experience with REST and SQL."

# ----- Optional -------------------------------------------------------------
# Set only if you use a non-default OpenAI-compatible endpoint (omit for api.openai.com).
# OPENAI_BASE_URL: str = "https://..."
VERBOSE_LOGGING: bool = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from esco_skill_extractor import ESCOOccupationMatcher, ESCOSkillExtractor, JobPosting

    matcher = ESCOOccupationMatcher(
        llm_provider="openai",
        llm_model_name=LLM_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
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
        openai_api_key=OPENAI_API_KEY,
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
    if OPENAI_API_KEY.startswith("sk-YOUR_"):
        sys.exit("Edit OPENAI_API_KEY in examples/02_openai_api.py")
    main()
