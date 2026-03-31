from esco_skill_extractor.models import JobPosting, MappedSkill


def test_mapped_skill_to_dict() -> None:
    s = MappedSkill(
        esco_skill_uri="http://example.org/skill/1",
        preferred_label="Python",
        similarity_score=0.9,
        category="essential",
        source="discovered",
    )
    d = s.to_dict()
    assert d["preferred_label"] == "Python"
    assert d["similarity_score"] == 0.9


def test_job_posting_optional_fields() -> None:
    j = JobPosting(title="Dev", esco_occupation_name="software developer")
    assert j.description is None
