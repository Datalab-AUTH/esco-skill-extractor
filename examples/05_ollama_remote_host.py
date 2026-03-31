"""
Example 5 — Ollama on another machine (or non-default port), **direct Ollama HTTP API**.

This uses ``llm_provider="ollama"`` and ``ollama_host`` only. There is **no** Open WebUI and **no**
``openai_base_url``: traffic goes straight to Ollama’s ``/api/chat``.

On the server, pull the model once: ``ollama pull <LLM_MODEL_NAME>``. Ensure the host/port is
reachable (firewall, VPN, etc.).

For Ollama on this machine, prefer ``01_ollama_local.py`` (or set ``OLLAMA_HOST`` below to
``http://127.0.0.1:11434``).

Set ``OLLAMA_HOST`` below to ``None`` to rely on the ``OLLAMA_HOST`` **environment variable**
instead.

Prerequisites: ``pip install -e .``

Run::

    python examples/05_ollama_remote_host.py
"""

from __future__ import annotations

import json
import logging

# ----- Required -------------------------------------------------------------
# Base URL of the Ollama server (no ``/v1`` path — not an OpenAI-compatible proxy).
# Use None to read from the OLLAMA_HOST environment variable (same as the Ollama CLI).
OLLAMA_HOST: str | None = "YOUR_HOST:11434"

LLM_MODEL_NAME: str = "deepseek-r1:7b"
EMBEDDING_MODEL: str = "nomic"

JOB_TITLE: str = "Room attendant"
JOB_DESCRIPTION: str = """Cleaning rooms and common areas"""
JOB_QUALIFICATIONS: str = "1+ year hotel experience"

# ----- Optional -------------------------------------------------------------
VERBOSE_LOGGING: bool = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from esco_skill_extractor import ESCOOccupationMatcher, ESCOSkillExtractor, JobPosting

    if OLLAMA_HOST is None:
        host = None
    else:
        host = OLLAMA_HOST.strip() or None

    matcher = ESCOOccupationMatcher(
        llm_provider="ollama",
        llm_model_name=LLM_MODEL_NAME,
        ollama_host=host,
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
        ollama_host=host,
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
