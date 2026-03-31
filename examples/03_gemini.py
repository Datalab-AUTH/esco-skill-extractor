"""
Example 3 — Google Gemini for LLM steps (embeddings still local ``nomic`` by default).

Prerequisites::

    pip install -e ".[gemini]"

Get an API key: https://aistudio.google.com/apikey

Run::

    python examples/03_gemini.py
"""

from __future__ import annotations

import json
import logging
import sys

# ----- Required -------------------------------------------------------------
GOOGLE_API_KEY: str = "YOUR_GEMINI_API_KEY"  # change me

# Gemini model id (see Google AI docs, e.g. gemini-2.0-flash, gemini-1.5-flash)
LLM_MODEL_NAME: str = "gemini-2.0-flash"

EMBEDDING_MODEL: str = "nomic"

JOB_TITLE: str = "Data analyst"
JOB_DESCRIPTION: str = "SQL dashboards, stakeholder reporting, Python notebooks."
JOB_QUALIFICATIONS: str = "Strong SQL; visualization tools; analytical mindset."

# ----- Optional -------------------------------------------------------------
VERBOSE_LOGGING: bool = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from esco_skill_extractor import ESCOOccupationMatcher, ESCOSkillExtractor, JobPosting

    matcher = ESCOOccupationMatcher(
        llm_provider="gemini",
        llm_model_name=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
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
        llm_provider="gemini",
        llm_model_name=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
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
    if GOOGLE_API_KEY.startswith("YOUR_"):
        sys.exit(
            "Edit GOOGLE_API_KEY in examples/03_gemini.py. "
            'Install: pip install -e ".[gemini]"'
        )
    main()
