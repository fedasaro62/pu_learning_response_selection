from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gen_jobs import MAX_CONCURRENT_ARRAY_ELEMENTS, PLAN, TASKS


JOB_FILES = ("estimate_prior.job", "train_main.job", "train_estimated.job", "equivalence.job", "submit_all.sh")
COMMITTED = Path(__file__).parent.parent / "jobs"


def _array_bound_and_configs(text: str) -> tuple[int, list[str]]:
    import re

    bound = int(re.search(r"^#SBATCH --array=0-(\d+)(?:%\d+)?$", text, re.M).group(1))
    block = re.search(r"^CONFIGS=\(\n(.*?)\n\)$", text, re.M | re.S).group(1)
    return bound, block.splitlines()


def test_expected_training_config_count() -> None:
    """verify that PLAN and TASKS imply 24 training runs."""
    expected_count = sum(len(seeds) * len(TASKS) for _, _, _, seeds in PLAN)
    assert expected_count == 24


def test_generated_files_exist(tmp_path, monkeypatch) -> None:
    import gen_jobs

    monkeypatch.chdir(tmp_path)
    gen_jobs.main()
    assert {p.name for p in (tmp_path / "jobs").iterdir()} == set(JOB_FILES)


def test_array_bounds_match_configs() -> None:
    for name in ("estimate_prior.job", "train_main.job", "train_estimated.job"):
        bound, configs = _array_bound_and_configs((COMMITTED / name).read_text())
        assert bound == len(configs) - 1, name
    assert "--array" not in (COMMITTED / "equivalence.job").read_text()


def test_training_arrays_cover_the_whole_plan() -> None:
    total = sum(len(_array_bound_and_configs((COMMITTED / n).read_text())[1])
                for n in ("train_main.job", "train_estimated.job"))
    assert total == sum(len(seeds) * len(TASKS) for _, _, _, seeds in PLAN) == 24


def test_estimated_prior_only_in_estimated_array() -> None:
    _, main = _array_bound_and_configs((COMMITTED / "train_main.job").read_text())
    _, est = _array_bound_and_configs((COMMITTED / "train_estimated.job").read_text())
    assert not any("estimated" in c for c in main)
    assert all("estimated" in c for c in est)
    assert len(set(main + est)) == len(main) + len(est)


def test_jobs_cd_one_level_up_and_print_config() -> None:
    for name in ("estimate_prior.job", "train_main.job", "train_estimated.job", "equivalence.job"):
        text = (COMMITTED / name).read_text()
        assert 'cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"' in text, name
    for name in ("estimate_prior.job", "train_main.job", "train_estimated.job"):
        text = (COMMITTED / name).read_text()
        assert text.index('echo "config: ') < text.index("scripts/"), name


def test_plan_row_tags_are_unique() -> None:
    """verify that each row tag appears exactly once in PLAN."""
    tags = [row for row, _, _, _ in PLAN]
    assert len(tags) == len(set(tags)), f"duplicate row tags in PLAN: {tags}"


def test_jobs_activate_venv_through_shell_expansion() -> None:
    """verify that the venv location is a runtime expansion, never a committed path."""
    for name in ("estimate_prior.job", "train_main.job", "train_estimated.job"):
        assert 'source "${PU_VENV:-.venv}/bin/activate"' in (COMMITTED / name).read_text(), name
    assert "${PU_VENV:-.venv}/bin/python" in (COMMITTED.parent / "scripts" / "check_equivalence.sh").read_text()


def test_generated_files_leak_no_identity() -> None:
    import gen_jobs

    for name, text in gen_jobs.generate_jobs().items():
        assert "mail-user" not in text and "mail-type" not in text, name
        assert "/nfs/" not in text, name
    for name in JOB_FILES:
        text = (COMMITTED / name).read_text()
        assert "mail-user" not in text and "/nfs/" not in text, name


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

    body = gen_jobs.render_submit_script()
    assert "((" not in body.replace("$((", "")
    arithmetic = [line.strip() for line in body.splitlines() if line.strip().startswith("submitted=$((")]
    assert len(arithmetic) == 3
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail; submitted=0\n" + "\n".join(arithmetic) + "\necho $submitted"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0 and result.stdout.strip() == "3"


def test_arrays_are_throttled() -> None:
    import re

    for name in ("estimate_prior.job", "train_main.job", "train_estimated.job"):
        text = (COMMITTED / name).read_text()
        lines = re.findall(r"^#SBATCH --array=.*$", text, re.M)
        assert len(lines) == 1, name
        assert re.fullmatch(rf"#SBATCH --array=0-\d+%{MAX_CONCURRENT_ARRAY_ELEMENTS}", lines[0]), lines[0]
    assert MAX_CONCURRENT_ARRAY_ELEMENTS == 4


def test_equivalence_job_has_no_array_line() -> None:
    assert "--array" not in (COMMITTED / "equivalence.job").read_text()


def test_submit_script_does_not_override_the_throttle() -> None:
    sbatch_lines = [line for line in (COMMITTED / "submit_all.sh").read_text().splitlines() if "sbatch" in line]
    assert sbatch_lines
    assert not any("--array" in line or " -a " in line for line in sbatch_lines)
