"""Map job text to ESCO occupations and extract ESCO skills."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("esco-skill-extractor")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "ESCOOccupationMatcher",
    "ESCOSkillExtractor",
    "ExtractedSkill",
    "JobPosting",
    "MappedSkill",
    "bundled_esco_csv",
    "__version__",
]

if TYPE_CHECKING:
    from .models import ExtractedSkill, JobPosting, MappedSkill
    from .occupation import ESCOOccupationMatcher
    from .paths import bundled_esco_csv
    from .skill_extraction import ESCOSkillExtractor


def __getattr__(name: str) -> Any:
    if name == "ExtractedSkill":
        from .models import ExtractedSkill

        return ExtractedSkill
    if name == "JobPosting":
        from .models import JobPosting

        return JobPosting
    if name == "MappedSkill":
        from .models import MappedSkill

        return MappedSkill
    if name == "ESCOOccupationMatcher":
        from .occupation import ESCOOccupationMatcher

        return ESCOOccupationMatcher
    if name == "ESCOSkillExtractor":
        from .skill_extraction import ESCOSkillExtractor

        return ESCOSkillExtractor
    if name == "bundled_esco_csv":
        from .paths import bundled_esco_csv

        return bundled_esco_csv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
