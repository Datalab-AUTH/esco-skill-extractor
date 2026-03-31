"""
Example 6 — **Open WebUI** (OpenAI-compatible API) with **every constructor keyword** set
explicitly.

Use this as a checklist when wiring ``ESCOOccupationMatcher`` and ``ESCOSkillExtractor``.
Arguments that do not apply to Open WebUI + local embeddings are still passed (``None`` /
``False``) so you can see the full surface area.

Prerequisites: ``pip install -e .``

Run::

    python examples/06_openwebui_all_constructor_args.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from esco_skill_extractor.paths import bundled_esco_csv

# ----- Open WebUI (LLM via openai client) -----------------------------------
OPENWEBUI_API_KEY: str = "sk-YOUR_OPENWEBUI_KEY"
OPENWEBUI_BASE_URL: str = "https://YOUR_HOST/api/v1"
LLM_MODEL_NAME: str = "llama3.2"

# ----- Embeddings & data paths ----------------------------------------------
EMBEDDING_MODEL: str = "nomic"
# Separate cache dir so this example does not clash with other runs
EMBEDDINGS_CACHE_DIR: Path = Path.cwd() / "embeddings_cache_example06"

# Explicit bundled CSV paths (same as passing ``None`` for each path)
OCCUPATIONS_CSV: Path = Path(bundled_esco_csv("esco_occupations.csv"))
SKILLS_CSV: Path = Path(bundled_esco_csv("esco_skills.csv"))
OCCUPATION_SKILLS_MAPPING_CSV: Path = Path(bundled_esco_csv("occupation_skills_mapping.csv"))

# ----- ESCOOccupationMatcher-only -------------------------------------------
USE_LLM_VALIDATION: bool = True
DEFAULT_TOP_K: int = 5
DEFAULT_MIN_SIMILARITY: float = 0.5
DEFAULT_CLEAN_WITH_LLM: bool = True
FORCE_RECOMPUTE_OCCUPATION_EMBEDDINGS: bool = False

# ----- ESCOSkillExtractor-only ----------------------------------------------
SIMILARITY_THRESHOLD: float = 0.6
FORCE_RECOMPUTE_SKILL_EMBEDDINGS: bool = False

# ----- Shared ----------------------------------------------------------------
VERBOSE_LOGGING: bool = True

# Not used when ``llm_provider="openai"`` (set when using Ollama or Gemini)
OLLAMA_HOST: str | None = None
GOOGLE_API_KEY: str | None = None

JOB_TITLE: str = "Clinical nurse"
JOB_DESCRIPTION: str = "Patient care, medication administration, care planning."
JOB_QUALIFICATIONS: str = "RN license; hospital experience."


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from esco_skill_extractor import ESCOOccupationMatcher, ESCOSkillExtractor, JobPosting

    matcher = ESCOOccupationMatcher(
        occupations_csv_path=OCCUPATIONS_CSV,
        embedding_model=EMBEDDING_MODEL,
        use_llm_validation=USE_LLM_VALIDATION,
        llm_provider="openai",
        llm_model_name=LLM_MODEL_NAME,
        ollama_host=OLLAMA_HOST,
        openai_api_key=OPENWEBUI_API_KEY,
        openai_base_url=OPENWEBUI_BASE_URL,
        google_api_key=GOOGLE_API_KEY,
        embeddings_cache_dir=EMBEDDINGS_CACHE_DIR,
        force_recompute_embeddings=FORCE_RECOMPUTE_OCCUPATION_EMBEDDINGS,
        verbose_logging=VERBOSE_LOGGING,
        default_top_k=DEFAULT_TOP_K,
        default_min_similarity=DEFAULT_MIN_SIMILARITY,
        default_clean_with_llm=DEFAULT_CLEAN_WITH_LLM,
    )
    matches = matcher.find_best_occupation(
        job_title=JOB_TITLE,
        description=JOB_DESCRIPTION,
        qualifications=JOB_QUALIFICATIONS,
    )
    matcher.print_results(matches, JOB_TITLE)
    top = matches[0]["occupation_name"] if matches else JOB_TITLE

    extractor = ESCOSkillExtractor(
        skills_csv_path=SKILLS_CSV,
        occupations_csv_path=OCCUPATIONS_CSV,
        occupation_skills_mapping_csv_path=OCCUPATION_SKILLS_MAPPING_CSV,
        embedding_model=EMBEDDING_MODEL,
        llm_provider="openai",
        llm_model_name=LLM_MODEL_NAME,
        ollama_host=OLLAMA_HOST,
        similarity_threshold=SIMILARITY_THRESHOLD,
        openai_api_key=OPENWEBUI_API_KEY,
        openai_base_url=OPENWEBUI_BASE_URL,
        google_api_key=GOOGLE_API_KEY,
        embeddings_cache_dir=EMBEDDINGS_CACHE_DIR,
        force_recompute_embeddings=FORCE_RECOMPUTE_SKILL_EMBEDDINGS,
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
    if (
        OPENWEBUI_API_KEY.startswith("sk-YOUR_")
        or "YOUR_HOST" in OPENWEBUI_BASE_URL
    ):
        sys.exit(
            "Edit OPENWEBUI_API_KEY and OPENWEBUI_BASE_URL in "
            "examples/06_openwebui_all_constructor_args.py"
        )
    main()
