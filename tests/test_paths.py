from pathlib import Path

from esco_skill_extractor.paths import bundled_esco_csv, package_root


def test_package_root_exists() -> None:
    root = package_root()
    assert root.is_dir()
    assert (root / "data").is_dir()


def test_bundled_esco_csv_files_exist() -> None:
    for name in (
        "esco_skills.csv",
        "esco_occupations.csv",
        "occupation_skills_mapping.csv",
    ):
        path = bundled_esco_csv(name)
        assert Path(path).is_file(), path
