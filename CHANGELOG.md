# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-31

### Added

- Initial PyPI-oriented package layout (`src/esco_skill_extractor`).
- `ESCOOccupationMatcher` and `ESCOSkillExtractor` with bundled ESCO CSV data.
- **Explicit constructor arguments** for all runtime settings (no `.env` / environment-based `Config`).
- CLI: `esco-skill-extractor occupation|skills` and `python -m esco_skill_extractor`.
- Lazy imports in the top-level package to avoid loading heavy ML stacks until needed.
- `bundled_esco_csv()` helper for paths to packaged CSV files.
- `NOTICE` for ESCO data attribution and licensing pointers.

### Removed

- `Config` class, `python-dotenv`, and `.env.example` (configure via code or CLI flags).

### Changed

- Remote Ollama: `ollama_host` on matchers/extractor and `--ollama-host` on the CLI.
- Ollama LLM calls use **httpx** against `/api/chat` instead of the **`ollama`** Python package, avoiding a slow or stuck `import ollama` on Windows (the package builds a default client with `platform.machine()` / WMI at import time).
- Declared **`einops`** and **`httpx`**; dropped the **`ollama`** dependency.

### Added

- `llm_provider="gemini"` with `google_api_key` (optional extra `pip install datalab-esco-skill-extractor[gemini]`).
- `openai_base_url` for OpenAI-compatible APIs (e.g. Open WebUI); CLI `--openai-base-url`, `--google-api-key`.
- Example scripts: `examples/01_ollama_local.py`, `02_openai_api.py`, `03_gemini.py`, `04_openwebui_openai_compatible.py`, `05_ollama_remote_host.py`, `06_openwebui_all_constructor_args.py`.
