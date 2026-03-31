# esco-skill-extractor

[![CI](https://github.com/Datalab-AUTH/esco-skill-extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/Datalab-AUTH/esco-skill-extractor/actions/workflows/ci.yml)

**esco-skill-extractor** is a Python library that connects free-text job postings to [ESCO](https://esco.ec.europa.eu/en) (European skills and occupations) in two steps:

1. **Occupation matching** — embed the job text and rank ESCO occupations by semantic similarity, with optional LLM-based cleaning of noisy listings and optional LLM re-ranking of the top candidates.
2. **Skill extraction** — call an LLM to list skills implied by the posting, then map each phrase to ESCO skill concepts using embeddings, prioritising skills that are already linked to the chosen occupation in ESCO.

Everything is configured **in code** (constructor arguments and method parameters). The library does **not** use `.env` files. For Ollama, pass **`ollama_host="http://your-server:11434"`** to point at a remote server; if you omit it, the library uses the **`OLLAMA_HOST`** environment variable when set, otherwise `http://127.0.0.1:11434` (via direct HTTP to Ollama’s API, not the `ollama` Python package, so imports stay fast on Windows).

---

## Installation

Install **from this repository** (works before or without a PyPI release):

```bash
git clone https://github.com/Datalab-AUTH/esco-skill-extractor.git
cd esco-skill-extractor
pip install -e .
```

Or a one-line install from GitHub:

```bash
pip install "git+https://github.com/Datalab-AUTH/esco-skill-extractor.git"
```

After the project is published on PyPI, you can use:

```bash
pip install esco-skill-extractor
```

**PyTorch** is a dependency (CPU wheels from PyPI by default). For GPU builds, follow [PyTorch’s install guide](https://pytorch.org/get-started/locally/) for your platform.

**Development** (editable install + linters/tests):

```bash
git clone https://github.com/Datalab-AUTH/esco-skill-extractor.git
cd esco-skill-extractor
pip install -e ".[dev]"
```

---

## What ships in the package

Under `esco_skill_extractor/data/` the wheel includes three CSV extracts used by default:

| File | Role |
|------|------|
| `esco_occupations.csv` | Occupation URI, preferred label, description |
| `esco_skills.csv` | Skill URI, labels, description |
| `occupation_skills_mapping.csv` | Links occupations to essential/optional skills |

To use **your own** ESCO dumps, pass explicit paths into the constructors (see below). ESCO content is subject to the [ESCO legal notice](https://esco.ec.europa.eu/en/about-esco/legal-notice); see also `NOTICE` in this repository.

---

## Quick start (library)

### 1. Occupation matcher

```python
from esco_skill_extractor import ESCOOccupationMatcher

matcher = ESCOOccupationMatcher(
    # occupations_csv_path=None  -> bundled esco_occupations.csv
    llm_provider="ollama",
    llm_model_name="deepseek-r1:7b",
    embedding_model="nomic",
    verbose_logging=False,
)

matches = matcher.find_best_occupation(
    job_title="Room attendant",
    description="Cleaning guest rooms and public areas in a hotel.",
    qualifications="One year of housekeeping experience; English spoken.",
    # top_k / min_similarity / clean_with_llm omit -> use matcher defaults
)
for m in matches:
    print(m["occupation_name"], m["similarity_score"])
```

### 2. Skill extractor (after you know the occupation label)

Use the `preferredLabel` of the occupation (string match against the occupations table), typically the top match from step 1.

```python
from esco_skill_extractor import ESCOSkillExtractor, JobPosting

extractor = ESCOSkillExtractor(
    llm_provider="ollama",
    llm_model_name="deepseek-r1:7b",
    similarity_threshold=0.6,
)

job = JobPosting(
    title="Room attendant",
    esco_occupation_name="room attendant",
    description="…",
    qualifications="…",
)

skills = extractor.extract_skills(job)
for s in skills:
    print(s.preferred_label, s.similarity_score, s.source)  # predefined | discovered
```

### Ollama on a remote server

Run Ollama on a machine you control (or behind a reverse proxy), expose port **11434** (or your custom port), then pass the base URL to both classes:

```python
matcher = ESCOOccupationMatcher(
    llm_provider="ollama",
    llm_model_name="deepseek-r1:7b",
    ollama_host="http://192.168.1.50:11434",  # or https://ollama.example.com
    embedding_model="nomic",
)
```

Ensure the model is **pulled on that server** (`ollama pull …` on the host). The CLI accepts the same URL: `--ollama-host http://192.168.1.50:11434`.

### OpenAI instead of Ollama

Pass `llm_provider="openai"`, `openai_api_key="sk-…"`, and choose an OpenAI chat model for `llm_model_name`. If you set `embedding_model="openai"`, the same key is used for `text-embedding-3-small`.

Optional **`openai_base_url`** targets any **OpenAI-compatible** HTTP API (for example [Open WebUI](https://github.com/open-webui/open-webui) in front of Ollama: use your Open WebUI API key and base URL such as `https://your-host/api/v1`).

### Gemini (Google AI)

Use `llm_provider="gemini"`, `google_api_key="…"` (from [Google AI Studio](https://aistudio.google.com/apikey)), and a Gemini model id for `llm_model_name` (e.g. `gemini-2.0-flash`). Install the extra dependency:

```bash
# from a clone
pip install -e ".[gemini]"

# from PyPI, once published
pip install "esco-skill-extractor[gemini]"
```

---

## Example scripts (`examples/`)

Run from the repository root after `pip install -e .` (and `pip install -e ".[gemini]"` for the Gemini example):

| Script | Purpose |
|--------|---------|
| `01_ollama_local.py` | Ollama on localhost (optional `OLLAMA_HOST` for another host/port). |
| `02_openai_api.py` | Official OpenAI API with your key. |
| `03_gemini.py` | Google Gemini for LLM calls. |
| `04_openwebui_openai_compatible.py` | Open WebUI OpenAI-compatible URL + API key (Ollama behind Open WebUI). |
| `05_ollama_remote_host.py` | Ollama on a remote host/port via **direct** Ollama API (`ollama_host`); no Open WebUI. |
| `06_openwebui_all_constructor_args.py` | Open WebUI + **every** matcher/extractor constructor keyword set explicitly (reference). |

```bash
python examples/01_ollama_local.py
```

---

## API reference (constructors)

### `ESCOOccupationMatcher`

All parameters are **keyword-only** (after `*`).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `occupations_csv_path` | bundled file | Path to ESCO occupations CSV |
| `embedding_model` | `"nomic"` | Short alias (`nomic`, `bge-base`, …), full HuggingFace Sentence-Transformers id, or `"openai"` |
| `use_llm_validation` | `False` | Re-rank multiple high-similarity hits with the LLM |
| `llm_provider` | `"ollama"` | `"ollama"`, `"openai"`, or `"gemini"` |
| `llm_model_name` | `"deepseek-r1:7b"` | Provider-specific model id |
| `ollama_host` | `None` | Ollama base URL; if `None`, uses `OLLAMA_HOST` or `http://127.0.0.1:11434` |
| `openai_api_key` | `None` | **Required** if provider is OpenAI or embeddings are OpenAI; also used for OpenAI-compatible APIs |
| `openai_base_url` | `None` | OpenAI-compatible API base URL (e.g. Open WebUI) |
| `google_api_key` | `None` | **Required** if `llm_provider="gemini"` |
| `embeddings_cache_dir` | `./embeddings_cache` | Cache for occupation embedding matrix |
| `force_recompute_embeddings` | `False` | Ignore cache and rebuild |
| `verbose_logging` | `False` | Set this module’s logger to DEBUG |
| `default_top_k` | `5` | Default `top_k` in `find_best_occupation` when you omit it |
| `default_min_similarity` | `0.5` | Default cosine similarity floor `[0, 1]` |
| `default_clean_with_llm` | `True` | Default for stripping marketing text before matching |

`find_best_occupation(..., top_k=None, min_similarity=None, clean_with_llm=None)` uses those defaults when an argument is omitted.

### `ESCOSkillExtractor`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `skills_csv_path` | bundled | ESCO skills CSV |
| `occupations_csv_path` | bundled | ESCO occupations CSV |
| `occupation_skills_mapping_csv_path` | bundled | Occupation–skill relation file |
| `embedding_model` | `"nomic"` | Same conventions as the matcher |
| `llm_provider` / `llm_model_name` | `"ollama"` / `"deepseek-r1:7b"` | Used for JSON skill extraction |
| `ollama_host` | `None` | Same as occupation matcher |
| `openai_api_key` / `openai_base_url` | `None` | Same as occupation matcher |
| `google_api_key` | `None` | Required for Gemini |
| `similarity_threshold` | `0.6` | Minimum similarity to accept a mapped skill |
| `embeddings_cache_dir` | `./embeddings_cache` | Skill embedding cache |
| `force_recompute_embeddings` | `False` | Rebuild skill embeddings |
| `verbose_logging` | `False` | DEBUG logs for this module |

Useful methods: `extract_skills(job: JobPosting)`, `export_to_dataframe`, `set_similarity_threshold`, `clear_cache`, `get_cache_info`.

### `bundled_esco_csv(filename: str) -> str`

Returns the absolute path to a packaged data file, e.g. `bundled_esco_csv("esco_skills.csv")`, for logging or custom pipelines.

### Lazy imports

`import esco_skill_extractor` does **not** import PyTorch or Sentence Transformers until you access `ESCOOccupationMatcher`, `ESCOSkillExtractor`, or import their submodules. Lightweight helpers such as `bundled_esco_csv` live in `esco_skill_extractor.paths`.

---

## Command-line interface

The console script mirrors the library defaults; flags override constructors.

```text
esco-skill-extractor occupation --title "Data scientist" --description "SQL, Python, ML"
esco-skill-extractor skills --title "Room attendant" --occupation "room attendant" --description "…"
```

Common options (both subcommands):

- `--embedding-model`, `--llm-provider` (`ollama` \| `openai` \| `gemini`), `--llm-model`, `--ollama-host`, `--openai-api-key`, `--openai-base-url`, `--google-api-key`
- `--embeddings-cache-dir`, `--force-recompute-embeddings`, `--verbose`

`occupation` only:

- `--occupations-csv`, `--use-llm-validation`, `--default-top-k`, `--default-min-similarity`
- `--no-clean-with-llm` (turns off LLM cleaning default)
- `--top-k`, `--min-similarity` (override defaults for that invocation)
- `--save-csv`, `--output-dir`

`skills` only:

- `--skills-csv`, `--occupations-csv`, `--mapping-csv`, `--similarity-threshold`
- `--save-csv`, `--output-dir`

Equivalent: `python -m esco_skill_extractor …`

---

## Embedding model aliases

Both classes support the same built-in aliases (case-insensitive):

`nomic`, `bge-base`, `bge-large`, `bge-small`, `minilm`, `mpnet`

Any other string is passed through to Sentence Transformers as a HuggingFace model id. Use `openai` for OpenAI embeddings (requires `openai_api_key`).

---

## Logging

The library uses the standard `logging` module (`logging.getLogger("esco_skill_extractor.…")`). It does **not** call `logging.basicConfig` except in the CLI. In your application, configure handlers and levels as usual; set `verbose_logging=True` on the matcher/extractor if you want DEBUG messages from that component.

---

## Development

```bash
python -m ruff check src tests
python -m pytest
python -m build   # artefacts in dist/
```

---

## Publishing to PyPI

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Create a GitHub **Release** (tag). `.github/workflows/publish.yml` uploads via [trusted publishing](https://docs.pypi.org/trusted-publishers/) once the project is registered on PyPI.
3. Alternatively: `python -m build` and `twine upload dist/*`.

---

## Requirements

- Python 3.10+
- Running **Ollama** (or valid OpenAI credentials) when using LLM features
- Disk space and RAM for embedding models (first run may download weights from Hugging Face or OpenAI)

---

## License

MIT for this project’s **code** (`LICENSE`). ESCO taxonomy data remains under the Commission’s terms; see `NOTICE`.
