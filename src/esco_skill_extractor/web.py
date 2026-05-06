"""FastAPI web app exposing the occupation matcher and skill extractor.

Long-running matcher/extractor calls are dispatched as background jobs so the
UI can survive page refreshes, stream logs while a task runs, and replay past
results from a server-side history.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# ContextVar set inside each worker thread so the per-job log handler knows
# which job's buffer to write into.
_current_job_id: ContextVar[str | None] = ContextVar("current_job_id", default=None)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class _BaseConfig(BaseModel):
    embedding_model: str = "nomic"
    llm_provider: str = Field("ollama", pattern="^(ollama|openai|gemini)$")
    llm_model: str = "deepseek-r1:7b"
    ollama_host: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    google_api_key: str | None = None
    embeddings_cache_dir: str | None = None
    force_recompute_embeddings: bool = False
    verbose: bool = False


class OccupationRequest(_BaseConfig):
    title: str
    description: str | None = None
    qualifications: str | None = None
    use_llm_validation: bool = False
    clean_with_llm: bool = True
    top_k: int = 5
    min_similarity: float = 0.5


class SkillsRequest(_BaseConfig):
    title: str
    occupation: str
    description: str | None = None
    qualifications: str | None = None
    similarity_threshold: float = 0.6


def _config_key(req: _BaseConfig, kind: str) -> tuple:
    return (
        kind,
        req.embedding_model,
        req.llm_provider,
        req.llm_model,
        req.ollama_host or "",
        req.openai_api_key or "",
        req.openai_base_url or "",
        req.google_api_key or "",
        req.embeddings_cache_dir or "",
        bool(req.force_recompute_embeddings),
    )


class _InstanceCache:
    """Cache one matcher/extractor per unique configuration to avoid re-loading
    embeddings on every request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple, Any] = {}

    def get_occupation_matcher(self, req: OccupationRequest):
        from .occupation import ESCOOccupationMatcher

        key = _config_key(req, "occupation")
        with self._lock:
            inst = self._cache.get(key)
            if inst is None:
                logger.info("Building new ESCOOccupationMatcher")
                inst = ESCOOccupationMatcher(
                    embedding_model=req.embedding_model,
                    llm_provider=req.llm_provider,
                    llm_model_name=req.llm_model,
                    ollama_host=req.ollama_host,
                    openai_api_key=req.openai_api_key,
                    openai_base_url=req.openai_base_url,
                    google_api_key=req.google_api_key,
                    embeddings_cache_dir=req.embeddings_cache_dir,
                    force_recompute_embeddings=req.force_recompute_embeddings,
                    verbose_logging=req.verbose,
                )
                self._cache[key] = inst
            return inst

    def get_skill_extractor(self, req: SkillsRequest):
        from .skill_extraction import ESCOSkillExtractor

        key = _config_key(req, "skills") + (req.similarity_threshold,)
        with self._lock:
            inst = self._cache.get(key)
            if inst is None:
                logger.info("Building new ESCOSkillExtractor")
                inst = ESCOSkillExtractor(
                    embedding_model=req.embedding_model,
                    llm_provider=req.llm_provider,
                    llm_model_name=req.llm_model,
                    ollama_host=req.ollama_host,
                    similarity_threshold=req.similarity_threshold,
                    openai_api_key=req.openai_api_key,
                    openai_base_url=req.openai_base_url,
                    google_api_key=req.google_api_key,
                    embeddings_cache_dir=req.embeddings_cache_dir,
                    force_recompute_embeddings=req.force_recompute_embeddings,
                    verbose_logging=req.verbose,
                )
                self._cache[key] = inst
            return inst


# --------------------------------------------------------------------------- #
# Job model + store
# --------------------------------------------------------------------------- #

class Job:
    def __init__(self, kind: str, title: str, inputs: dict | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.title = title
        self.inputs: dict = inputs or {}
        self.status: str = "queued"  # queued | running | done | error
        self.result: Any = None
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._lock = threading.Lock()
        self._logs: list[dict] = []
        self._log_counter = 0

    @classmethod
    def from_snapshot(cls, data: dict) -> "Job":
        """Reconstruct a Job from a previously-persisted snapshot."""
        j = cls(data.get("kind", "occupation"), data.get("title", ""), data.get("inputs") or {})
        j.id = data.get("id", j.id)
        j.status = data.get("status", "error")
        j.result = data.get("result")
        j.error = data.get("error")
        j.created_at = data.get("created_at") or time.time()
        j.started_at = data.get("started_at")
        j.finished_at = data.get("finished_at")
        logs = data.get("logs") or []
        j._logs = list(logs)
        j._log_counter = data.get("log_count") or len(logs)
        return j

    def append_log(self, level: str, message: str) -> None:
        with self._lock:
            entry = {
                "i": self._log_counter,
                "ts": time.time(),
                "level": level,
                "message": message,
            }
            self._log_counter += 1
            self._logs.append(entry)
            # Cap log buffer size; counter keeps moving so client `since` stays valid
            if len(self._logs) > 1000:
                self._logs = self._logs[-800:]

    def mark_running(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = time.time()

    def mark_done(self, result: Any) -> None:
        with self._lock:
            self.status = "done"
            self.result = result
            self.finished_at = time.time()

    def mark_error(self, err: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = err
            self.finished_at = time.time()

    def snapshot(self, *, include_logs: bool = False, log_since: int = 0) -> dict:
        with self._lock:
            d = {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "inputs": self.inputs,
                "status": self.status,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "log_count": self._log_counter,
            }
            if include_logs:
                d["logs"] = [entry for entry in self._logs if entry["i"] >= log_since]
            return d


class _Persistence:
    """Disk persistence for jobs and UI settings under ``web_settings/``."""

    def __init__(self, base_dir: Path) -> None:
        self.base = Path(base_dir)
        self.jobs_dir = self.base / "jobs"
        self.settings_path = self.base / "settings.json"
        self.base.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_job(self, job: Job) -> None:
        snap = job.snapshot(include_logs=True)
        path = self.jobs_dir / f"{job.id}.json"
        try:
            payload = json.dumps(snap, default=str, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to serialize job %s for persistence", job.id)
            return
        try:
            with self._lock:
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(path)
        except OSError:
            logger.exception("Failed to write job file %s", path)

    def delete_job(self, job_id: str) -> None:
        path = self.jobs_dir / f"{job_id}.json"
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete job file %s", path)

    def load_jobs(self) -> list[dict]:
        if not self.jobs_dir.is_dir():
            return []
        out: list[dict] = []
        for f in self.jobs_dir.glob("*.json"):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable job file %s", f)
        return out

    def get_settings(self) -> dict:
        if not self.settings_path.is_file():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read settings file %s", self.settings_path)
            return {}

    def save_settings(self, data: dict) -> None:
        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to serialize settings")
            return
        try:
            with self._lock:
                tmp = self.settings_path.with_suffix(".json.tmp")
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self.settings_path)
        except OSError:
            logger.exception("Failed to write settings file %s", self.settings_path)


class JobStore:
    def __init__(
        self,
        max_jobs: int = 50,
        persistence: _Persistence | None = None,
    ) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._max = max_jobs
        self._lock = threading.Lock()
        self._persistence = persistence

    def restore_from_disk(self) -> None:
        if not self._persistence:
            return
        loaded = self._persistence.load_jobs()
        loaded.sort(key=lambda d: d.get("created_at") or 0)
        for snap in loaded:
            job = Job.from_snapshot(snap)
            # A persisted job stuck in queued/running means the server died
            # mid-run; surface that to the UI rather than showing a fake spinner.
            if job.status in ("queued", "running"):
                job.status = "error"
                if not job.error:
                    job.error = "Interrupted (server restarted before completion)"
                if not job.finished_at:
                    job.finished_at = time.time()
                self._persistence.save_job(job)
            self._jobs[job.id] = job
            self._order.append(job.id)
        # Trim to cap, deleting the oldest from disk too.
        evicted: list[str] = []
        while len(self._order) > self._max:
            old = self._order.popleft()
            self._jobs.pop(old, None)
            evicted.append(old)
        for old in evicted:
            self._persistence.delete_job(old)

    def add(self, job: Job) -> None:
        evicted: list[str] = []
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._max:
                old = self._order.popleft()
                self._jobs.pop(old, None)
                evicted.append(old)
        if self._persistence:
            self._persistence.save_job(job)
            for old in evicted:
                self._persistence.delete_job(old)

    def persist(self, job: Job) -> None:
        if self._persistence:
            self._persistence.save_job(job)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_recent(self) -> list[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in reversed(self._order) if jid in self._jobs]


class _JobLogHandler(logging.Handler):
    """Routes log records to the active job's buffer based on a ContextVar."""

    def __init__(self, store: JobStore) -> None:
        super().__init__()
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        try:
            job_id = _current_job_id.get()
            if not job_id:
                return
            job = self.store.get(job_id)
            if not job:
                return
            try:
                msg = self.format(record)
            except Exception:
                msg = record.getMessage()
            job.append_log(record.levelname, msg)
        except Exception:
            # Never let logging crash the job
            pass


# --------------------------------------------------------------------------- #
# Job runners
# --------------------------------------------------------------------------- #

def _run_occupation_job(
    store: JobStore, cache: _InstanceCache, job: Job, req: OccupationRequest
) -> None:
    token = _current_job_id.set(job.id)
    try:
        job.mark_running()
        store.persist(job)
        logger.info("Starting occupation match for %r", req.title)
        matcher = cache.get_occupation_matcher(req)
        matcher.use_llm_validation = bool(req.use_llm_validation)
        matches = matcher.find_best_occupation(
            job_title=req.title,
            description=req.description or None,
            qualifications=req.qualifications or None,
            top_k=req.top_k,
            min_similarity=req.min_similarity,
            clean_with_llm=req.clean_with_llm,
        )
        logger.info("Occupation match complete: %d candidate(s)", len(matches))
        job.mark_done({"job_title": req.title, "matches": matches})
    except Exception as e:
        logger.exception("Occupation match failed")
        job.mark_error(f"{type(e).__name__}: {e}")
    finally:
        store.persist(job)
        _current_job_id.reset(token)


def _run_skills_job(
    store: JobStore, cache: _InstanceCache, job: Job, req: SkillsRequest
) -> None:
    from .models import JobPosting

    token = _current_job_id.set(job.id)
    try:
        job.mark_running()
        store.persist(job)
        logger.info("Starting skill extraction for %r (occupation=%r)", req.title, req.occupation)
        extractor = cache.get_skill_extractor(req)
        posting = JobPosting(
            title=req.title,
            esco_occupation_name=req.occupation,
            description=req.description or None,
            qualifications=req.qualifications or None,
        )
        mapped = extractor.extract_skills(posting)
        logger.info("Skill extraction complete: %d skill(s)", len(mapped))
        job.mark_done({
            "job_title": req.title,
            "occupation": req.occupation,
            "skills": [s.to_dict() for s in mapped],
        })
    except Exception as e:
        logger.exception("Skill extraction failed")
        job.mark_error(f"{type(e).__name__}: {e}")
    finally:
        store.persist(job)
        _current_job_id.reset(token)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #

def create_app(settings_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="ESCO Skill Extractor",
        description="Local API for matching ESCO occupations and extracting ESCO skills.",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    base_dir = Path(settings_dir) if settings_dir is not None else Path.cwd() / "web_settings"
    persistence = _Persistence(base_dir)
    logger.info("Persistence directory: %s", persistence.base.resolve())

    cache = _InstanceCache()
    store = JobStore(persistence=persistence)
    store.restore_from_disk()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="esco-job")

    # Attach the per-job log handler to namespaces we care about.
    handler = _JobLogHandler(store)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    for name in ("esco_skill_extractor", "sentence_transformers", "transformers"):
        target = logging.getLogger(name)
        if not any(isinstance(h, _JobLogHandler) for h in target.handlers):
            target.addHandler(handler)
        # Default level is WARNING — bump to INFO so the UI can see progress.
        # Submodules that explicitly call setLevel() will override this.
        if target.level == logging.NOTSET or target.level > logging.INFO:
            target.setLevel(logging.INFO)

    @app.on_event("shutdown")
    def _shutdown() -> None:  # pragma: no cover
        executor.shutdown(wait=False, cancel_futures=True)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/occupation/match")
    def submit_occupation(req: OccupationRequest) -> dict:
        inputs = {
            "title": req.title,
            "description": req.description,
            "qualifications": req.qualifications,
        }
        job = Job("occupation", req.title, inputs=inputs)
        store.add(job)
        executor.submit(_run_occupation_job, store, cache, job, req)
        return {"job_id": job.id}

    @app.post("/api/skills/extract")
    def submit_skills(req: SkillsRequest) -> dict:
        inputs = {
            "title": req.title,
            "description": req.description,
            "qualifications": req.qualifications,
            "occupation": req.occupation,
        }
        job = Job("skills", req.title, inputs=inputs)
        store.add(job)
        executor.submit(_run_skills_job, store, cache, job, req)
        return {"job_id": job.id}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [j.snapshot() for j in store.list_recent()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, since: int = 0) -> dict:
        j = store.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="Job not found")
        return j.snapshot(include_logs=True, log_since=since)

    @app.get("/api/settings")
    def get_settings() -> dict:
        return persistence.get_settings()

    @app.put("/api/settings")
    def save_settings(data: dict = Body(...)) -> dict:
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Settings must be a JSON object")
        persistence.save_settings(data)
        return {"status": "saved"}

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))

    return app


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    settings_dir: str | Path | None = None,
) -> None:
    """Start uvicorn. Used by the ``esco-skill-extractor web`` CLI."""
    import os

    import uvicorn

    if reload:
        # In reload mode uvicorn re-imports the factory, so pass the dir via env.
        if settings_dir is not None:
            os.environ["ESCO_WEB_SETTINGS_DIR"] = str(settings_dir)
        uvicorn.run(
            "esco_skill_extractor.web:_reload_factory",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
    else:
        uvicorn.run(create_app(settings_dir=settings_dir), host=host, port=port)


def _reload_factory() -> FastAPI:
    """Factory wrapper used only by ``--reload`` so uvicorn can re-import it."""
    import os

    return create_app(settings_dir=os.environ.get("ESCO_WEB_SETTINGS_DIR"))
