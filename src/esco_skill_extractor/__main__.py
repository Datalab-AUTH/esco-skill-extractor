"""CLI: ``python -m esco_skill_extractor`` or the ``esco-skill-extractor`` console script."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_embedding_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--embedding-model",
        default="nomic",
        help="Embedding backend: alias (nomic, bge-base, …) or HF model id, or openai",
    )
    p.add_argument(
        "--llm-provider",
        choices=("ollama", "openai", "gemini"),
        default="ollama",
    )
    p.add_argument("--llm-model", default="deepseek-r1:7b", help="LLM model id for the provider")
    p.add_argument(
        "--ollama-host",
        default=None,
        help='Ollama server URL (default: client uses OLLAMA_HOST env or http://127.0.0.1:11434)',
    )
    p.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI or OpenAI-compatible API key (Open WebUI, etc.)",
    )
    p.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible API base URL (e.g. Open WebUI https://host/api/v1)",
    )
    p.add_argument(
        "--google-api-key",
        default=None,
        help="Google AI (Gemini) API key when --llm-provider gemini",
    )
    p.add_argument(
        "--embeddings-cache-dir",
        default=None,
        help="Embedding cache directory (default: ./embeddings_cache)",
    )
    p.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore cache and rebuild embeddings",
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")


def _cmd_occupation(args: argparse.Namespace) -> int:
    from .occupation import ESCOOccupationMatcher

    matcher = ESCOOccupationMatcher(
        occupations_csv_path=args.occupations_csv,
        embedding_model=args.embedding_model,
        use_llm_validation=args.use_llm_validation,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model,
        ollama_host=args.ollama_host,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        google_api_key=args.google_api_key,
        embeddings_cache_dir=args.embeddings_cache_dir,
        force_recompute_embeddings=args.force_recompute_embeddings,
        verbose_logging=args.verbose,
        default_top_k=args.default_top_k,
        default_min_similarity=args.default_min_similarity,
        default_clean_with_llm=not args.no_clean_with_llm,
    )
    matches = matcher.find_best_occupation(
        job_title=args.title,
        description=args.description or None,
        qualifications=args.qualifications or None,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    matcher.print_results(matches, args.title)

    if args.save_csv and matches:
        import pandas as pd

        out = Path(args.output_dir) / "occupation_matches.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(matches).to_csv(out, index=False)
        logging.getLogger(__name__).info("Wrote %s", out)
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    from .models import JobPosting
    from .skill_extraction import ESCOSkillExtractor

    extractor = ESCOSkillExtractor(
        skills_csv_path=args.skills_csv,
        occupations_csv_path=args.occupations_csv,
        occupation_skills_mapping_csv_path=args.mapping_csv,
        embedding_model=args.embedding_model,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model,
        ollama_host=args.ollama_host,
        similarity_threshold=args.similarity_threshold,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        google_api_key=args.google_api_key,
        embeddings_cache_dir=args.embeddings_cache_dir,
        force_recompute_embeddings=args.force_recompute_embeddings,
        verbose_logging=args.verbose,
    )
    job = JobPosting(
        title=args.title,
        esco_occupation_name=args.occupation,
        description=args.description or None,
        qualifications=args.qualifications or None,
    )
    mapped = extractor.extract_skills(job)
    extractor.print_results(mapped, job.title)

    if args.save_csv and mapped:
        out = Path(args.output_dir) / "extracted_skills.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        extractor.export_to_dataframe(mapped).to_csv(out, index=False)
        logging.getLogger(__name__).info("Wrote %s", out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="esco-skill-extractor",
        description="Match jobs to ESCO occupations and extract ESCO-linked skills.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_occ = sub.add_parser("occupation", help="Match a job posting to ESCO occupations")
    _add_embedding_llm_args(p_occ)
    p_occ.add_argument("--title", required=True)
    p_occ.add_argument("--description", default="")
    p_occ.add_argument("--qualifications", default="")
    p_occ.add_argument(
        "--occupations-csv",
        default=None,
        help="Default: bundled esco_occupations.csv",
    )
    p_occ.add_argument("--use-llm-validation", action="store_true")
    p_occ.add_argument("--default-top-k", type=int, default=5)
    p_occ.add_argument("--default-min-similarity", type=float, default=0.5)
    p_occ.add_argument(
        "--no-clean-with-llm",
        action="store_true",
        help="Disable LLM cleaning by default (constructor default)",
    )
    p_occ.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override default_top_k for this run only",
    )
    p_occ.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Override default_min_similarity for this run",
    )
    p_occ.add_argument("--save-csv", action="store_true", help="Write occupation_matches.csv")
    p_occ.add_argument("--output-dir", default="output", help="Directory for CSV output")
    p_occ.set_defaults(func=_cmd_occupation)

    p_sk = sub.add_parser("skills", help="Extract ESCO skills for a job posting")
    _add_embedding_llm_args(p_sk)
    p_sk.add_argument("--title", required=True)
    p_sk.add_argument(
        "--occupation",
        required=True,
        help="ESCO occupation preferredLabel (e.g. from occupation matcher)",
    )
    p_sk.add_argument("--description", default="")
    p_sk.add_argument("--qualifications", default="")
    p_sk.add_argument("--skills-csv", default=None, help="Default: bundled esco_skills.csv")
    p_sk.add_argument(
        "--occupations-csv",
        default=None,
        help="Default: bundled esco_occupations.csv",
    )
    p_sk.add_argument(
        "--mapping-csv",
        default=None,
        help="Default: bundled occupation_skills_mapping.csv",
    )
    p_sk.add_argument("--similarity-threshold", type=float, default=0.6)
    p_sk.add_argument("--save-csv", action="store_true")
    p_sk.add_argument("--output-dir", default="output")
    p_sk.set_defaults(func=_cmd_skills)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
