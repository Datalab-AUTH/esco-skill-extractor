"""
ESCO Occupation Matcher with LLM-based Job Description Cleaning

Matches job postings to ESCO occupations using semantic similarity.

Author: SkillScapes Analytics
Version: 2.1 - Flexible embedding models and explicit constructor configuration
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .ollama_http import ollama_chat
from .paths import bundled_esco_csv

logger = logging.getLogger(__name__)


class ESCOOccupationMatcher:
    """
    Match job postings to ESCO occupations using semantic similarity.

    Features:
    - LLM-based job description cleaning (removes marketing, benefits, etc.)
    - Embedding-based semantic matching with flexible model support
    - Optional LLM validation of top matches
    - Memory-mapped embedding caching for fast loads
    - GPU support
    - Explicit constructor parameters (no environment variables)
    """

    # Predefined model aliases for convenience
    MODEL_ALIASES = {
        "nomic": "nomic-ai/nomic-embed-text-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
    }

    def __init__(
        self,
        *,
        occupations_csv_path: str | Path | None = None,
        embedding_model: str = "nomic",
        use_llm_validation: bool = False,
        llm_provider: str = "ollama",
        llm_model_name: str = "deepseek-r1:7b",
        ollama_host: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        google_api_key: str | None = None,
        embeddings_cache_dir: str | Path | None = None,
        force_recompute_embeddings: bool = False,
        verbose_logging: bool = False,
        default_top_k: int = 5,
        default_min_similarity: float = 0.5,
        default_clean_with_llm: bool = True,
    ) -> None:
        """
        Initialize the occupation matcher.

        Args:
            occupations_csv_path: CSV with ESCO occupations. If ``None``, uses the bundled
                ``esco_occupations.csv``.
            embedding_model: Short alias (e.g. ``nomic``, ``bge-base``) or a full
                HuggingFace Sentence-Transformers id, or ``openai`` for OpenAI embeddings.
            use_llm_validation: If True, optionally re-rank top embedding hits with the LLM.
            llm_provider: ``ollama``, ``openai``, or ``gemini`` (Gemini needs ``pip install google-generativeai``).
            llm_model_name: Model id for the chosen provider.
            ollama_host: Ollama server URL, e.g. ``http://192.168.1.10:11434``.
                If ``None``, uses ``OLLAMA_HOST`` or ``http://127.0.0.1:11434``.
            openai_api_key: Required when ``llm_provider == "openai"`` or when
                ``embedding_model`` is ``openai``. Also used for OpenAI-compatible APIs (e.g. Open WebUI).
            openai_base_url: Optional OpenAI API base URL (e.g. Open WebUI: ``https://host/api/v1``).
            google_api_key: Google AI (Gemini) API key when ``llm_provider == "gemini"``.
            embeddings_cache_dir: Directory for cached occupation embeddings. Defaults to
                ``./embeddings_cache`` under the current working directory.
            force_recompute_embeddings: Ignore cache and rebuild occupation embeddings.
            verbose_logging: If True, set this module's logger to DEBUG.
            default_top_k: Default number of occupation candidates when ``find_best_occupation``
                is called without ``top_k``.
            default_min_similarity: Default cosine similarity floor (0–1) for ``find_best_occupation``.
            default_clean_with_llm: Default for ``clean_with_llm`` in ``find_best_occupation``.
        """
        logger.setLevel(logging.DEBUG if verbose_logging else logging.INFO)

        self.occupations_csv_path = str(
            Path(occupations_csv_path) if occupations_csv_path is not None else Path(bundled_esco_csv("esco_occupations.csv"))
        )
        self.use_llm_validation = use_llm_validation
        self.llm_provider = llm_provider.strip().lower()
        self.llm_model_name = llm_model_name.strip()
        self.openai_api_key = (openai_api_key or "").strip() or None
        self.openai_base_url = (openai_base_url or "").strip() or None
        self.google_api_key = (google_api_key or "").strip() or None
        self._ollama_host: str | None = None
        if self.llm_provider == "ollama":
            h = (ollama_host or "").strip()
            self._ollama_host = h or None
        self._openai_client = None
        self._gemini_model = None
        self.embeddings_cache_dir = Path(embeddings_cache_dir) if embeddings_cache_dir is not None else Path.cwd() / "embeddings_cache"
        self.force_recompute = force_recompute_embeddings
        self.default_top_k = default_top_k
        self.default_min_similarity = default_min_similarity
        self.default_clean_with_llm = default_clean_with_llm

        self._validate_constructor_args()

        self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

        self.occupations_df = pd.read_csv(self.occupations_csv_path)
        logger.info("Loaded %s ESCO occupations", len(self.occupations_df))

        self.embedding_model_name = embedding_model
        self.resolved_model_name = self.MODEL_ALIASES.get(
            embedding_model.lower(),
            embedding_model,
        )

        if self.resolved_model_name.lower() == "openai" and not self.openai_api_key:
            raise ValueError("openai_api_key is required when embedding_model is openai")

        if self.openai_api_key and (
            self.llm_provider == "openai"
            or str(self.resolved_model_name).lower() == "openai"
        ):
            from openai import OpenAI

            client_kw: dict = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                client_kw["base_url"] = self.openai_base_url
            self._openai_client = OpenAI(**client_kw)

        if self.llm_provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as err:
                raise ImportError(
                    'Install google-generativeai for Gemini: pip install "google-generativeai>=0.8"'
                ) from err
            genai.configure(api_key=self.google_api_key)
            self._gemini_model = genai.GenerativeModel(self.llm_model_name)

        logger.info("Initializing embedding model: %s", embedding_model)
        self.embedder = self._initialize_embedder(self.resolved_model_name, self.openai_api_key)

        logger.info("Loading or creating occupation embeddings...")
        self.occupation_embeddings = self._load_or_create_occupation_embeddings()
        logger.info("Occupation matcher initialized successfully!")

    def _validate_constructor_args(self) -> None:
        if self.llm_provider not in {"ollama", "openai", "gemini"}:
            raise ValueError('llm_provider must be "ollama", "openai", or "gemini"')
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("openai_api_key is required when llm_provider is openai")
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("google_api_key is required when llm_provider is gemini")
        if not Path(self.occupations_csv_path).is_file():
            raise FileNotFoundError(f"occupations_csv_path not found: {self.occupations_csv_path}")
        if not 0.0 <= self.default_min_similarity <= 1.0:
            raise ValueError("default_min_similarity must be between 0.0 and 1.0")
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")

    def _initialize_embedder(self, model_name: str, api_key: str | None):
        """Initialize the selected embedding model"""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Special case for OpenAI
        if model_name.lower() == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required for openai embedding model")
            logger.info("Using OpenAI text-embedding-3-small")
            return "openai"

        # Try to load any SentenceTransformer model
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load embedding model '{model_name}': {e}")

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for text using selected model"""
        if self.resolved_model_name.lower() == "openai":
            assert self._openai_client is not None
            response = self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        else:
            # Check if model needs BGE prefix
            prefix = ""
            if "bge" in self.resolved_model_name.lower():
                prefix = "Represent this sentence for searching relevant passages: "
                texts = [prefix + text for text in texts]

            embeddings = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32
            )
            return embeddings

    def _get_occupation_embeddings_cache_path(self) -> Path:
        """Generate cache file path for occupation embeddings"""
        # Sanitize model name for filesystem
        safe_model_name = self.embedding_model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        cache_filename = f"esco_occupation_embeddings_{safe_model_name}.npy"
        return self.embeddings_cache_dir / cache_filename

    def _get_occupation_metadata_cache_path(self) -> Path:
        """Generate metadata cache path for occupations"""
        safe_model_name = self.embedding_model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        metadata_filename = f"esco_occupation_metadata_{safe_model_name}.pkl"
        return self.embeddings_cache_dir / metadata_filename

    def _load_or_create_occupation_embeddings(self) -> np.ndarray:
        """
        Load occupation embeddings from cache using memory mapping.
        If cache doesn't exist or is invalid, create and cache embeddings.
        """
        cache_path = self._get_occupation_embeddings_cache_path()
        metadata_path = self._get_occupation_metadata_cache_path()

        # Check if cache exists and is valid
        if not self.force_recompute and cache_path.exists() and metadata_path.exists():
            try:
                metadata = joblib.load(metadata_path)
                current_num_occupations = len(self.occupations_df)
                cached_num_occupations = metadata.get('num_occupations', 0)

                # Check if CSV was modified
                csv_mtime = os.path.getmtime(self.occupations_csv_path)
                cache_mtime = metadata.get('creation_time', 0)

                if (cached_num_occupations == current_num_occupations and
                    cache_mtime >= csv_mtime):
                    logger.info(f"Loading occupation embeddings from cache (memory-mapped): {cache_path}")
                    start_load = time.time()
                    embeddings = np.load(cache_path, mmap_mode='r')
                    load_time = time.time() - start_load
                    logger.info(f"Loaded {len(embeddings)} occupation embeddings in {load_time:.3f} seconds")
                    return embeddings
                else:
                    logger.info("Occupation cache is outdated, regenerating embeddings...")
            except Exception as e:
                logger.warning(f"Failed to load occupation cache: {e}. Regenerating...")

        # Create embeddings from scratch
        logger.info(f"Creating embeddings for {len(self.occupations_df)} ESCO occupations...")
        start_time = time.time()
        embeddings = self._create_occupation_embeddings()
        elapsed = time.time() - start_time
        logger.info(f"Created {len(embeddings)} occupation embeddings in {elapsed:.2f} seconds")

        # Save to cache
        try:
            logger.info(f"Saving occupation embeddings to cache: {cache_path}")
            np.save(cache_path, embeddings)

            metadata = {
                'num_occupations': len(self.occupations_df),
                'embedding_dim': embeddings.shape[1],
                'model_name': self.embedding_model_name,
                'resolved_model': self.resolved_model_name,
                'creation_time': time.time()
            }
            joblib.dump(metadata, metadata_path)

            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            logger.info(f"Occupation embeddings cached successfully ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.warning(f"Failed to cache occupation embeddings: {e}")

        return embeddings

    def _create_occupation_text(self, row) -> str:
        """Combine occupation info into searchable text"""
        text_parts = [row['preferredLabel']]

        if pd.notna(row.get('altLabels')):
            text_parts.append(str(row['altLabels']))

        if pd.notna(row.get('description')):
            desc = str(row['description'])[:500]
            text_parts.append(desc)

        return " | ".join(text_parts)

    def _create_occupation_embeddings(self) -> np.ndarray:
        """Create embeddings for all ESCO occupations"""
        occupation_texts = []
        for _, row in self.occupations_df.iterrows():
            text = self._create_occupation_text(row)
            occupation_texts.append(text)

        # Batch processing
        import torch
        if torch.cuda.is_available() and self.resolved_model_name.lower() != "openai":
            batch_size = 128
        elif self.resolved_model_name.lower() == "openai":
            batch_size = 100
        else:
            batch_size = 32

        logger.info(f"Using batch size: {batch_size}")

        all_embeddings = []
        for i in range(0, len(occupation_texts), batch_size):
            batch = occupation_texts[i:i + batch_size]
            try:
                batch_embeddings = self._embed_text(batch)
                all_embeddings.append(batch_embeddings)

                if (i // batch_size) % 5 == 0:
                    logger.info(
                        f"Processed {min(i + batch_size, len(occupation_texts))}/{len(occupation_texts)} occupations"
                    )
            except Exception as e:
                logger.error(f"Error processing batch {i}-{i + batch_size}: {e}")
                raise

        return np.vstack(all_embeddings)

    def _clean_job_description_with_llm(
        self,
        job_title: str,
        raw_description: str | None = None,
        raw_qualifications: str | None = None
    ) -> tuple[str, str]:
        """
        Use LLM to clean and extract relevant job information.
        Removes marketing content, company info, benefits, etc.
        """
        # Build input text
        text_parts = [f"Job Title: {job_title}"]
        if raw_description:
            text_parts.append(f"Description:\n{raw_description}")
        if raw_qualifications:
            text_parts.append(f"Qualifications:\n{raw_qualifications}")

        raw_text = "\n\n".join(text_parts)

        # Prompt for cleaning
        prompt = f"""You are a professional job description analyst. Your task is to extract and clarify the ESSENTIAL occupational information from a job listing, removing all irrelevant content.

INPUT JOB LISTING:
{raw_text}

INSTRUCTIONS:
1. EXTRACT the core job responsibilities and duties (what the person actually does day-to-day)
2. EXTRACT the actual job requirements and qualifications needed
3. REMOVE all of the following:
   - Company history, background, and marketing content
   - Benefits, perks, salary information, bonuses
   - Application instructions and contact information
   - Location details and working hours
   - Generic motivational language
   - Company statistics and metrics
   - Company values and philosophy statements
   - Employment type information
   - Reference numbers and job codes
4. DO NOT summarize or shorten the actual job duties - keep all relevant details
5. DO NOT add information that isn't in the original text

OUTPUT FORMAT - Return ONLY valid JSON:
{{
  "cleaned_description": "Clear description of what the job actually involves",
  "cleaned_qualifications": "Actual requirements and skills needed"
}}"""

        try:
            if self.llm_provider == "openai":
                assert self._openai_client is not None
                response = self._openai_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting occupational information from job listings."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                )
                llm_output = response.choices[0].message.content
            elif self.llm_provider == "gemini":
                assert self._gemini_model is not None
                import google.generativeai as genai

                gcfg = genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
                response = self._gemini_model.generate_content(prompt, generation_config=gcfg)
                llm_output = response.text
            else:
                llm_output = ollama_chat(
                    host=self._ollama_host,
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting occupational information from job listings."},
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    options={"temperature": 0.1},
                )

            # Parse response
            cleaned = json.loads(llm_output)
            cleaned_description = cleaned.get('cleaned_description', raw_description or '')
            cleaned_qualifications = cleaned.get('cleaned_qualifications', raw_qualifications or '')

            logger.info("Job description cleaned successfully")
            return cleaned_description, cleaned_qualifications

        except Exception as e:
            logger.error(f"Failed to clean job description with LLM: {e}")
            logger.info("Falling back to original description")
            return raw_description or '', raw_qualifications or ''

    def find_best_occupation(
        self,
        job_title: str,
        description: str | None = None,
        qualifications: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        clean_with_llm: bool | None = None
    ) -> list[dict]:
        """
        Find the best matching ESCO occupation(s) for a job posting.

        Args:
            job_title: Job title (required)
            description: Job description (optional)
            qualifications: Job qualifications (optional)
            top_k: Number of top matches (default: constructor ``default_top_k``).
            min_similarity: Minimum cosine similarity in ``[0, 1]`` (default: ``default_min_similarity``).
            clean_with_llm: Strip marketing text via LLM first (default: ``default_clean_with_llm``).

        Returns:
            List of occupation matches with scores
        """
        top_k = self.default_top_k if top_k is None else top_k
        min_similarity = self.default_min_similarity if min_similarity is None else min_similarity
        clean_with_llm = self.default_clean_with_llm if clean_with_llm is None else clean_with_llm

        # Clean description with LLM if enabled
        if clean_with_llm and (description or qualifications):
            logger.info("Cleaning job description with LLM...")
            description, qualifications = self._clean_job_description_with_llm(
                job_title, description, qualifications
            )

        # Create job posting embedding
        job_text_parts = [job_title]
        if description:
            job_text_parts.append(description)
        if qualifications:
            job_text_parts.append(qualifications)

        job_text = " ".join(job_text_parts)
        job_embedding = self._embed_text([job_text])

        # Compute similarities
        similarities = cosine_similarity(job_embedding, self.occupation_embeddings)[0]

        # Get top-K matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity < min_similarity:
                continue

            occupation = self.occupations_df.iloc[idx]

            esco_uri = occupation.get('occupation_uri', '')

            candidates.append({
                'esco_uri': esco_uri,
                'occupation_name': occupation['preferredLabel'],
                'similarity_score': round(similarity, 4),
                'description': occupation.get('description', ''),
                'isco_code': occupation.get('iscoGroup', '')
            })

        # Optional LLM validation
        if self.use_llm_validation and len(candidates) > 1:
            candidates = self._validate_with_llm(
                job_title, description, qualifications, candidates
            )

        return candidates

    def _validate_with_llm(
        self,
        job_title: str,
        description: str | None,
        qualifications: str | None,
        candidates: list[dict]
    ) -> list[dict]:
        """Use LLM to validate and rank top candidates"""
        candidates_text = "\n".join([
            f"{i + 1}. {c['occupation_name']} (similarity: {c['similarity_score']})"
            for i, c in enumerate(candidates)
        ])

        job_info = f"Job Title: {job_title}"
        if description:
            job_info += f"\nDescription: {description[:500]}..."
        if qualifications:
            job_info += f"\nQualifications: {qualifications[:300]}..."

        prompt = f"""Given this job posting:
{job_info}

Which of these ESCO occupations is the best match?
{candidates_text}

Return ONLY the number (1-{len(candidates)}) of the best match and a brief reason (max 20 words)."""

        try:
            if self.llm_provider == "openai":
                assert self._openai_client is not None
                response = self._openai_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                llm_output = response.choices[0].message.content
            elif self.llm_provider == "gemini":
                assert self._gemini_model is not None
                import google.generativeai as genai

                gcfg = genai.GenerationConfig(temperature=0.1)
                response = self._gemini_model.generate_content(prompt, generation_config=gcfg)
                llm_output = response.text
            else:
                llm_output = ollama_chat(
                    host=self._ollama_host,
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                )

            match = re.search(r'\b([1-9])\b', llm_output)
            if match:
                chosen_idx = int(match.group(1)) - 1
                if 0 <= chosen_idx < len(candidates):
                    chosen = candidates[chosen_idx]
                    chosen['llm_validated'] = True
                    chosen['llm_reason'] = llm_output
                    return [chosen] + [c for i, c in enumerate(candidates) if i != chosen_idx]

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")

        return candidates

    def clear_cache(self):
        """Clear the occupation embeddings cache"""
        cache_path = self._get_occupation_embeddings_cache_path()
        metadata_path = self._get_occupation_metadata_cache_path()

        if cache_path.exists():
            os.remove(cache_path)
            logger.info(f"Removed cache: {cache_path}")

        if metadata_path.exists():
            os.remove(metadata_path)
            logger.info(f"Removed metadata: {metadata_path}")

    def get_cache_info(self) -> dict:
        """Get cache information"""
        cache_path = self._get_occupation_embeddings_cache_path()
        metadata_path = self._get_occupation_metadata_cache_path()

        info = {
            'cache_exists': cache_path.exists(),
            'cache_path': str(cache_path),
            'metadata_path': str(metadata_path)
        }

        if cache_path.exists():
            info['cache_size_mb'] = os.path.getsize(cache_path) / (1024 * 1024)

        if metadata_path.exists():
            try:
                metadata = joblib.load(metadata_path)
                info['metadata'] = metadata
            except Exception:
                pass

        return info

    def print_results(self, matches: list[dict], job_title: str):
        """Print results in standardized format"""
        print("\n" + "=" * 80)
        print(f"OCCUPATION MATCHING RESULTS: {job_title}")
        print("=" * 80)

        if not matches:
            print("\nNo matches found above similarity threshold.")
        else:
            for i, match in enumerate(matches, 1):
                print(f"\n{i}. {match['occupation_name']}")
                print(f"   Similarity Score: {match['similarity_score']}")
                print(f"   ESCO URI: {match['esco_uri']}")
                print(f"   ISCO Code: {match.get('isco_code', 'N/A')}")
                if 'llm_validated' in match:
                    print(f"   LLM Validated: {match['llm_reason']}")

        print("\n" + "=" * 80)
        print(f"Total matches: {len(matches)}")
        if matches:
            print(f"Top match: {matches[0]['occupation_name']} (score: {matches[0]['similarity_score']})")
        print("=" * 80 + "\n")

    def get_match(self, matches: list[dict], job_title: str):
        if not matches:
            return None
        else:
            other_matches = {}
            for i, match in enumerate(matches[1:], 1):
                other_matches['match_'+str(i)]={'match':match['occupation_name'],'score':match['similarity_score'],'esco_uri':match['esco_uri']}
            return {'match':matches[0]['occupation_name'],'score':matches[0]['similarity_score'],'esco_uri':matches[0]['esco_uri'],'other_matches':other_matches}
