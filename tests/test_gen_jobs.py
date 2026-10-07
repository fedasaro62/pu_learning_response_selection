from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gen_jobs import PLAN, TASKS


def test_expected_file_count() -> None:
    """verify that PLAN and TASKS produce the expected number of job files."""
    expected_count = sum(len(seeds) * len(TASKS) for _, _, _, seeds in PLAN)
    assert expected_count == 18, (
        f"PLAN implies {expected_count} files, not 18; "
        f"check that sum({[len(seeds) for _, _, _, seeds in PLAN]} * {len(TASKS)}) == 18"
    )


def test_plan_row_tags_are_unique() -> None:
    """verify that each row tag appears exactly once in PLAN."""
    tags = [row for row, _, _, _ in PLAN]
    assert len(tags) == len(set(tags)), f"duplicate row tags in PLAN: {tags}"


def test_venv_path_exists() -> None:
    """verify that the configured venv path exists."""
    from gen_jobs import VENV_PATH
    venv_activate = Path(VENV_PATH)
    assert venv_activate.exists(), (
        f"venv not found at {VENV_PATH}; jobs would fail at startup"
    )


def test_rpu_is_consistent() -> None:
    """verify that RPU matches the hardcoded value in finetune.sh template."""
    from gen_jobs import RPU
    assert RPU == 0.8, f"RPU should be 0.8, not {RPU}"
