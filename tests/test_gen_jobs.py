from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gen_jobs import PLAN, TASKS


def test_expected_file_count() -> None:
    """verify that PLAN and TASKS produce the expected number of job files."""
    expected_count = sum(len(seeds) * len(TASKS) for _, _, _, seeds in PLAN)
    assert expected_count == 16, (
        f"PLAN implies {expected_count} files, not 16; "
        f"check that sum({[len(seeds) for _, _, _, seeds in PLAN]} * {len(TASKS)}) == 16"
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


def test_committed_jobs_match_the_generator(tmp_path, monkeypatch):
    import filecmp
    import gen_jobs

    monkeypatch.chdir(tmp_path)
    gen_jobs.main()
    committed = Path(__file__).parent.parent / "jobs"

    def files(root: Path) -> set[Path]:
        return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}

    assert files(tmp_path / "jobs") == files(committed)
    stale = [f for f in files(committed) if not filecmp.cmp(tmp_path / "jobs" / f, committed / f, shallow=False)]
    assert not stale, f"committed jobs differ from generator output, rerun scripts/gen_jobs.py: {stale}"


def test_submit_script_counters_survive_set_e(tmp_path):
    import subprocess

    import gen_jobs

    monkey_written = [(Path("a.job"), "r", "ubuntu", "pu", "zero", 1)] * 3
    body = gen_jobs._generate_submit_script(monkey_written, set(), PLAN, TASKS)
    assert "((" not in body.replace("$((", "")
    arithmetic = [line for line in body.splitlines() if "submitted=$((" in line]
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail; submitted=0\n" + "\n".join(arithmetic) + "\necho $submitted"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0 and result.stdout.strip() == "3"
