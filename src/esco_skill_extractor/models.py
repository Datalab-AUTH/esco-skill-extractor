"""Data models for job postings and mapped skills."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JobPosting:
    title: str
    esco_occupation_name: str
    description: str | None = None
    qualifications: str | None = None


@dataclass
class ExtractedSkill:
    skill_text: str
    category: str


@dataclass
class MappedSkill:
    esco_skill_uri: str
    preferred_label: str
    similarity_score: float
    category: str
    source: str

    def to_dict(self) -> dict:
        return {
            "esco_skill_uri": self.esco_skill_uri,
            "preferred_label": self.preferred_label,
            "similarity_score": self.similarity_score,
            "category": self.category,
            "source": self.source,
        }
