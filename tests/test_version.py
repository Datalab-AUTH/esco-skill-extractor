import esco_skill_extractor


def test_version_is_string() -> None:
    assert isinstance(esco_skill_extractor.__version__, str)
    assert len(esco_skill_extractor.__version__) >= 3
