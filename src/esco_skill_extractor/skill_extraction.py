"""
ESCO Skill Extraction Module

Extracts ESCO skills from job descriptions using embeddings and LLM-based skill identification.

Author: SkillScapes Analytics
Version: 2.1 - Flexible embedding models and explicit constructor configuration
"""

import json
import logging
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ExtractedSkill, JobPosting, MappedSkill
from .ollama_http import ollama_chat
from .paths import bundled_esco_csv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

logger = logging.getLogger(__name__)


class ESCOSkillExtractor:
    """
    Main class for extracting and mapping ESCO skills from job descriptions.

    Features:
    - Multiple embedding models (any HuggingFace model or OpenAI)
    - Multiple LLM providers (Ollama, OpenAI)
    - Persistent caching of embeddings with memory-mapped loading
    - Two-pass skill matching (predefined + discovered)
    - Configurable similarity thresholds
    - Occupation lookup by name or URI
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
        skills_csv_path: str | Path | None = None,
        occupations_csv_path: str | Path | None = None,
        occupation_skills_mapping_csv_path: str | Path | None = None,
        embedding_model: str = "nomic",
        llm_provider: str = "ollama",
        llm_model_name: str = "deepseek-r1:7b",
        ollama_host: str | None = None,
        similarity_threshold: float = 0.6,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        google_api_key: str | None = None,
        embeddings_cache_dir: str | Path | None = None,
        force_recompute_embeddings: bool = False,
        verbose_logging: bool = False,
    ) -> None:
        """
        Initialize the ESCO skill extractor.

        Args:
            skills_csv_path: ESCO skills CSV. If ``None``, uses bundled ``esco_skills.csv``.
            occupations_csv_path: ESCO occupations CSV. If ``None``, uses bundled file.
            occupation_skills_mapping_csv_path: Occupation–skill relations. If ``None``, uses bundled file.
            embedding_model: Alias (e.g. ``nomic``) or full HuggingFace model id, or ``openai``.
            llm_provider: ``ollama``, ``openai``, or ``gemini``.
            llm_model_name: Model name for that provider.
            ollama_host: Ollama server URL. If ``None``, uses ``OLLAMA_HOST`` or
                ``http://127.0.0.1:11434``.
            similarity_threshold: Minimum cosine similarity (0–1) when mapping text to ESCO skills.
            openai_api_key: Required if ``llm_provider == "openai"`` or ``embedding_model == "openai"``.
            openai_base_url: OpenAI-compatible API base URL (e.g. Open WebUI).
            google_api_key: Gemini API key when ``llm_provider == "gemini"``.
            embeddings_cache_dir: Cache directory for skill embeddings (default: ``./embeddings_cache``).
            force_recompute_embeddings: Ignore cache and rebuild skill embeddings.
            verbose_logging: Enable DEBUG logging for this module.
        """
        logger.setLevel(logging.DEBUG if verbose_logging else logging.INFO)

        self.skills_csv_path = str(
            Path(skills_csv_path) if skills_csv_path is not None else Path(bundled_esco_csv("esco_skills.csv"))
        )
        self.occupations_csv_path = str(
            Path(occupations_csv_path)
            if occupations_csv_path is not None
            else Path(bundled_esco_csv("esco_occupations.csv"))
        )
        self.occupation_skills_mapping_csv_path = str(
            Path(occupation_skills_mapping_csv_path)
            if occupation_skills_mapping_csv_path is not None
            else Path(bundled_esco_csv("occupation_skills_mapping.csv"))
        )

        self.similarity_threshold = similarity_threshold
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
        self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_recompute = force_recompute_embeddings

        self._validate_constructor_args()

        logger.info("Loading ESCO data...")
        self.skills_df = pd.read_csv(self.skills_csv_path)
        self.occupations_df = pd.read_csv(self.occupations_csv_path)
        self.occupation_skills_df = pd.read_csv(self.occupation_skills_mapping_csv_path)

        logger.info("Loaded %s skills, %s occupations", len(self.skills_df), len(self.occupations_df))

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

        logger.info("Loading or creating embeddings for all ESCO skills...")
        self.skill_embeddings = self._load_or_create_embeddings()
        logger.info("Initialization complete.")

    def _validate_constructor_args(self) -> None:
        if self.llm_provider not in {"ollama", "openai", "gemini"}:
            raise ValueError('llm_provider must be "ollama", "openai", or "gemini"')
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("openai_api_key is required when llm_provider is openai")
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("google_api_key is required when llm_provider is gemini")
        for label, path_str in (
            ("skills_csv_path", self.skills_csv_path),
            ("occupations_csv_path", self.occupations_csv_path),
            ("occupation_skills_mapping_csv_path", self.occupation_skills_mapping_csv_path),
        ):
            if not Path(path_str).is_file():
                raise FileNotFoundError(f"{label} not found: {path_str}")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    def _initialize_embedder(self, model_name: str, api_key: str | None):
        """Initialize the selected embedding model"""
        import torch

        # Auto-detect GPU
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
            # Sentence transformers models
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

    def _get_embeddings_cache_path(self) -> Path:
        """Generate cache file path based on model name"""
        safe_model_name = self.embedding_model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        cache_filename = f"esco_skill_embeddings_{safe_model_name}.npy"
        return self.embeddings_cache_dir / cache_filename

    def _get_metadata_cache_path(self) -> Path:
        """Generate metadata cache path"""
        safe_model_name = self.embedding_model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        metadata_filename = f"esco_skill_metadata_{safe_model_name}.pkl"
        return self.embeddings_cache_dir / metadata_filename

    def _load_or_create_embeddings(self) -> np.ndarray:
        """
        Load embeddings from cache using memory mapping for faster access.
        """
        cache_path = self._get_embeddings_cache_path()
        metadata_path = self._get_metadata_cache_path()

        # Check if cache exists and is valid
        if not self.force_recompute and cache_path.exists() and metadata_path.exists():
            try:
                # Load metadata to verify cache validity
                metadata = joblib.load(metadata_path)

                # Verify the cache matches current data
                current_num_skills = len(self.skills_df)
                cached_num_skills = metadata.get('num_skills', 0)

                # Check if skills CSV was modified after cache creation
                skills_mtime = os.path.getmtime(self.skills_csv_path)
                cache_mtime = metadata.get('creation_time', 0)

                if (cached_num_skills == current_num_skills and
                    cache_mtime >= skills_mtime):
                    logger.info(f"Loading embeddings from cache (memory-mapped): {cache_path}")
                    start_load = time.time()

                    # Load with memory mapping for instant access (~10x faster)
                    embeddings = np.load(cache_path, mmap_mode='r')
                    load_time = time.time() - start_load
                    logger.info(f"Loaded {len(embeddings)} embeddings in {load_time:.3f} seconds (memory-mapped)")
                    return embeddings
                else:
                    logger.info("Cache is outdated, regenerating embeddings...")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Regenerating embeddings...")

        # Create embeddings from scratch
        logger.info("Creating embeddings from scratch (this may take a few minutes)...")
        start_time = time.time()
        embeddings = self._create_skill_embeddings()
        elapsed = time.time() - start_time
        logger.info(f"Created {len(embeddings)} embeddings in {elapsed:.2f} seconds")

        # Save to cache
        try:
            logger.info(f"Saving embeddings to cache: {cache_path}")
            np.save(cache_path, embeddings)

            # Save metadata
            metadata = {
                'num_skills': len(self.skills_df),
                'embedding_dim': embeddings.shape[1],
                'model_name': self.embedding_model_name,
                'resolved_model': self.resolved_model_name,
                'creation_time': time.time()
            }
            joblib.dump(metadata, metadata_path)

            # Log file size
            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            logger.info(f"Embeddings cached successfully ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

        return embeddings

    def _create_skill_embeddings(self) -> np.ndarray:
        """Create embeddings for all ESCO skills using description + labels"""
        skill_texts = []
        for _, row in self.skills_df.iterrows():
            # Combine preferredLabel, altLabels, and description
            text_parts = [row['preferredLabel']]

            if pd.notna(row.get('altLabels')):
                text_parts.append(str(row['altLabels']))

            if pd.notna(row.get('description')):
                text_parts.append(str(row['description']))

            combined_text = " | ".join(text_parts)
            skill_texts.append(combined_text)

        # Larger batch size for GPU
        import torch
        if torch.cuda.is_available() and self.resolved_model_name.lower() != "openai":
            batch_size = 128
        elif self.resolved_model_name.lower() == "openai":
            batch_size = 100
        else:
            batch_size = 32

        logger.info(f"Using batch size: {batch_size}")

        all_embeddings = []
        for i in range(0, len(skill_texts), batch_size):
            batch = skill_texts[i:i + batch_size]
            try:
                batch_embeddings = self._embed_text(batch)
                all_embeddings.append(batch_embeddings)

                if (i // batch_size) % 10 == 0:
                    logger.info(f"Processed {min(i + batch_size, len(skill_texts))}/{len(skill_texts)} skills")
            except Exception as e:
                logger.error(f"Error processing batch {i}-{i + batch_size}: {e}")
                raise

        return np.vstack(all_embeddings)

    def _get_predefined_skills_by_name(self, esco_occupation_name: str) -> dict[str, list[str]]:
        """
        Get predefined essential and optional skills for an ESCO occupation by name.
        """
        # Find the occupation URI from the occupation name
        occupation_match = self.occupations_df[
            self.occupations_df['preferredLabel'].str.lower() == esco_occupation_name.lower()
        ]

        if occupation_match.empty:
            logger.warning(f"No exact occupation match found for: '{esco_occupation_name}'")
            logger.info("Attempting fuzzy match...")

            # Try partial match
            occupation_match = self.occupations_df[
                self.occupations_df['preferredLabel'].str.contains(
                    esco_occupation_name,
                    case=False,
                    na=False
                )
            ]

            if occupation_match.empty:
                logger.warning(f"No occupation found matching: '{esco_occupation_name}'")
                return {'essential': [], 'optional': []}
            elif len(occupation_match) > 1:
                logger.warning(f"Multiple occupations found matching '{esco_occupation_name}':")
                for idx, row in occupation_match.iterrows():
                    logger.warning(f"  - {row['preferredLabel']}")
                logger.info(f"Using first match: {occupation_match.iloc[0]['preferredLabel']}")

        esco_occupation_uri = occupation_match.iloc[0]['occupation_uri']
        occupation_name_found = occupation_match.iloc[0]['preferredLabel']
        logger.info(f"Found occupation: '{occupation_name_found}' -> {esco_occupation_uri}")

        # Now get skills using the URI
        return self._get_predefined_skills(esco_occupation_uri)

    def _get_predefined_skills(self, esco_occupation_uri: str) -> dict[str, list[str]]:
        """
        Get predefined essential and optional skills for an ESCO occupation by URI.
        """
        # Flexible column name detection
        occupation_col = None
        skill_col = None
        relation_col = None

        # Find occupation column
        for col in self.occupation_skills_df.columns:
            if 'occupation' in col.lower() and 'uri' in col.lower():
                occupation_col = col
                break
            elif col.lower() in ['concepturi', 'occupation_id', 'esco_occupation']:
                occupation_col = col
                break

        # Find skill column
        for col in self.occupation_skills_df.columns:
            if 'skill' in col.lower() and 'uri' in col.lower():
                skill_col = col
                break
            elif col.lower() in ['skill_uri', 'skill_id', 'esco_skill']:
                skill_col = col
                break

        # Find relation type column
        for col in self.occupation_skills_df.columns:
            if 'relation' in col.lower() or 'type' in col.lower():
                relation_col = col
                break
            elif col.lower() in ['relationtype', 'skill_type', 'essentiality']:
                relation_col = col
                break

        if not occupation_col or not skill_col:
            raise ValueError(
                f"Could not find required columns in occupation_skills_mapping CSV. "
                f"Available columns: {list(self.occupation_skills_df.columns)}. "
                f"Expected columns containing 'occupation' and 'skill' with 'uri'."
            )

        logger.debug(f"Using columns: occupation={occupation_col}, skill={skill_col}, relation={relation_col}")

        # Filter for this occupation
        occupation_skills = self.occupation_skills_df[
            self.occupation_skills_df[occupation_col] == esco_occupation_uri
        ]

        if relation_col and relation_col in occupation_skills.columns:
            # If we have relation type info
            essential_skills = occupation_skills[
                occupation_skills[relation_col].str.lower().isin(['essential', 'required', 'mandatory'])
            ][skill_col].tolist()

            optional_skills = occupation_skills[
                occupation_skills[relation_col].str.lower().isin(['optional', 'preferred', 'recommended'])
            ][skill_col].tolist()
        else:
            # No relation type column - treat all as essential
            essential_skills = occupation_skills[skill_col].tolist()
            optional_skills = []

        logger.info(f"Found {len(essential_skills)} essential and {len(optional_skills)} optional predefined skills")

        return {
            'essential': essential_skills,
            'optional': optional_skills
        }

    def _extract_skills_with_llm(self, job_posting: JobPosting) -> list[ExtractedSkill]:
        """Use LLM to extract raw skills from job description"""
        # Construct prompt with clear sections
        text_parts = []
        text_parts.append(f"JOB TITLE: {job_posting.title}")
        if job_posting.description:
            text_parts.append(f"JOB DESCRIPTION:\n{job_posting.description}")
        if job_posting.qualifications:
            text_parts.append(f"QUALIFICATIONS/REQUIREMENTS:\n{job_posting.qualifications}")

        job_text = "\n\n".join(text_parts)

        # Structured prompt with step-by-step instructions
        prompt = f"""You are a professional HR skills extractor. Read the job posting below and extract every skill, ability, knowledge area, and competency required.

{job_text}

INSTRUCTIONS:
1. Read the ENTIRE job posting carefully (title, description, duties, qualifications)
2. For EACH duty listed, extract the skill needed to perform it
3. For EACH qualification, extract the specific skill or knowledge
4. Include technical skills, soft skills, knowledge areas, and abilities
5. DO NOT include: benefits, salary, perks, company offerings, work environment descriptions
6. Use clear, professional skill names

CATEGORIZATION:
- Mark as "essential" if: explicitly required, mandatory, or part of core duties
- Mark as "optional" if: preferred, nice-to-have, or described as a plus

OUTPUT FORMAT - Return ONLY valid JSON with this exact structure:
{{
  "essential": [
    "skill name 1",
    "skill name 2",
    "skill name 3"
  ],
  "optional": [
    "optional skill 1",
    "optional skill 2"
  ]
}}

Make sure to extract ALL skills mentioned. Be thorough and complete."""

        # Call LLM
        try:
            if self.llm_provider == "ollama":
                llm_output = ollama_chat(
                    host=self._ollama_host,
                    model=self.llm_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional skills extraction assistant. Extract ALL skills from job postings comprehensively. Return only valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    format="json",
                    options={"temperature": 0.1, "top_p": 0.9},
                )
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
                assert self._openai_client is not None
                response = self._openai_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional skills extraction assistant. Extract ALL skills from job postings comprehensively."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                )
                llm_output = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

        # Parse response
        try:
            skills_dict = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {llm_output}")
            raise ValueError(f"Invalid JSON from LLM: {e}")

        extracted_skills = []
        for skill in skills_dict.get('essential', []):
            extracted_skills.append(ExtractedSkill(skill, "essential"))
        for skill in skills_dict.get('optional', []):
            extracted_skills.append(ExtractedSkill(skill, "optional"))

        logger.info(f"Extracted {len(extracted_skills)} raw skills from LLM")

        return extracted_skills

    def _map_skill_to_esco(
        self,
        raw_skill: str,
        predefined_skill_uris: list[str]
    ) -> tuple[str, float, bool] | None:
        """
        Map a raw skill to ESCO skill using embeddings.

        Returns:
            Tuple of (esco_skill_uri, similarity_score, is_predefined) or None
        """
        # Embed the raw skill
        raw_skill_embedding = self._embed_text([raw_skill])

        # First pass: Check against predefined skills
        if predefined_skill_uris:
            predefined_indices = self.skills_df[
                self.skills_df['skill_uri'].isin(predefined_skill_uris)
            ].index.tolist()

            if predefined_indices:
                predefined_embeddings = self.skill_embeddings[predefined_indices]
                predefined_similarities = cosine_similarity(
                    raw_skill_embedding,
                    predefined_embeddings
                )[0]

                max_idx = np.argmax(predefined_similarities)
                max_similarity = predefined_similarities[max_idx]

                if max_similarity >= self.similarity_threshold:
                    matched_skill_uri = self.skills_df.iloc[predefined_indices[max_idx]]['skill_uri']
                    return (matched_skill_uri, float(max_similarity), True)

        # Second pass: Check against all ESCO skills
        all_similarities = cosine_similarity(raw_skill_embedding, self.skill_embeddings)[0]
        max_idx = np.argmax(all_similarities)
        max_similarity = all_similarities[max_idx]

        if max_similarity >= self.similarity_threshold:
            matched_skill_uri = self.skills_df.iloc[max_idx]['skill_uri']
            return (matched_skill_uri, float(max_similarity), False)

        return None

    def extract_skills(self, job_posting: JobPosting) -> list[MappedSkill]:
        """
        Main method to extract and map ESCO skills from a job posting.

        Args:
            job_posting: JobPosting object with ESCO occupation name and job details

        Returns:
            List of MappedSkill objects
        """
        logger.info(f"Processing job: {job_posting.title}")

        # Get predefined skills for this occupation BY NAME
        predefined_skills = self._get_predefined_skills_by_name(job_posting.esco_occupation_name)
        all_predefined = predefined_skills['essential'] + predefined_skills['optional']

        logger.info(f"Found {len(predefined_skills['essential'])} predefined essential skills, "
                   f"{len(predefined_skills['optional'])} optional skills")

        # Extract raw skills using LLM
        extracted_skills = self._extract_skills_with_llm(job_posting)

        # Map each extracted skill to ESCO
        mapped_skills = []
        matched_uris = set()

        for extracted_skill in extracted_skills:
            # Determine which predefined skills to prioritize based on category
            if extracted_skill.category == "essential":
                priority_predefined = predefined_skills['essential']
            else:
                priority_predefined = all_predefined

            mapping_result = self._map_skill_to_esco(
                extracted_skill.skill_text,
                priority_predefined
            )

            if mapping_result:
                esco_uri, similarity, is_predefined = mapping_result

                # Avoid duplicates
                if esco_uri not in matched_uris:
                    matched_uris.add(esco_uri)

                    skill_label = self.skills_df[
                        self.skills_df['skill_uri'] == esco_uri
                    ]['preferredLabel'].iloc[0]

                    mapped_skills.append(MappedSkill(
                        esco_skill_uri=esco_uri,
                        preferred_label=skill_label,
                        similarity_score=similarity,
                        category=extracted_skill.category,
                        source="predefined" if is_predefined else "discovered"
                    ))

                    logger.debug(f"Mapped '{extracted_skill.skill_text}' -> '{skill_label}' "
                               f"(similarity: {similarity:.3f}, source: {'predefined' if is_predefined else 'discovered'})")
            else:
                logger.debug(f"No match found for skill: '{extracted_skill.skill_text}'")

        logger.info(f"Mapped {len(mapped_skills)} ESCO skills "
                   f"({sum(1 for s in mapped_skills if s.source == 'predefined')} predefined, "
                   f"{sum(1 for s in mapped_skills if s.source == 'discovered')} discovered)")

        return mapped_skills

    def export_to_dataframe(self, mapped_skills: list[MappedSkill]) -> pd.DataFrame:
        """Export mapped skills to a pandas DataFrame"""
        return pd.DataFrame([skill.to_dict() for skill in mapped_skills])

    def set_similarity_threshold(self, threshold: float):
        """Update the similarity threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.similarity_threshold = threshold
        logger.info(f"Similarity threshold updated to {threshold}")

    def clear_cache(self):
        """Clear the embeddings cache"""
        cache_path = self._get_embeddings_cache_path()
        metadata_path = self._get_metadata_cache_path()

        if cache_path.exists():
            os.remove(cache_path)
            logger.info(f"Removed embeddings cache: {cache_path}")

        if metadata_path.exists():
            os.remove(metadata_path)
            logger.info(f"Removed metadata cache: {metadata_path}")

    def get_cache_info(self) -> dict:
        """Get information about the current cache"""
        cache_path = self._get_embeddings_cache_path()
        metadata_path = self._get_metadata_cache_path()

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

    def print_results(self, mapped_skills: list[MappedSkill], job_title: str):
        """Print results in standardized format"""
        print("\n" + "=" * 80)
        print(f"SKILL EXTRACTION RESULTS: {job_title}")
        print("=" * 80)

        if not mapped_skills:
            print("\nNo skills mapped.")
        else:
            # Group by category
            essential_skills = [s for s in mapped_skills if s.category == "essential"]
            optional_skills = [s for s in mapped_skills if s.category == "optional"]

            if essential_skills:
                print(f"\n📌 ESSENTIAL SKILLS ({len(essential_skills)}):")
                for i, skill in enumerate(essential_skills, 1):
                    source_icon = "⭐" if skill.source == "predefined" else "🔍"
                    print(f"   {i}. {skill.preferred_label} {source_icon}")
                    print(f"      Similarity: {skill.similarity_score:.4f} | Source: {skill.source}")
                    print(f"      URI: {skill.esco_skill_uri}")

            if optional_skills:
                print(f"\n✨ OPTIONAL SKILLS ({len(optional_skills)}):")
                for i, skill in enumerate(optional_skills, 1):
                    source_icon = "⭐" if skill.source == "predefined" else "🔍"
                    print(f"   {i}. {skill.preferred_label} {source_icon}")
                    print(f"      Similarity: {skill.similarity_score:.4f} | Source: {skill.source}")
                    print(f"      URI: {skill.esco_skill_uri}")

        print("\n" + "=" * 80)
        print(f"Total skills: {len(mapped_skills)}")
        print(f"Essential: {sum(1 for s in mapped_skills if s.category == 'essential')}")
        print(f"Optional: {sum(1 for s in mapped_skills if s.category == 'optional')}")
        print(f"Predefined: {sum(1 for s in mapped_skills if s.source == 'predefined')}")
        print(f"Discovered: {sum(1 for s in mapped_skills if s.source == 'discovered')}")
        if mapped_skills:
            avg_sim = np.mean([s.similarity_score for s in mapped_skills])
            print(f"Average similarity: {avg_sim:.3f}")
        print("=" * 80 + "\n")


