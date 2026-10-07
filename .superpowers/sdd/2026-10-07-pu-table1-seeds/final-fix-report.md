# Final fix wave report

All commands used /nfs/home/dasaro/research/innonext-BERT_FP/.venv/bin/python. Full suite: 67 passed (was 56).

- C1: tools/estimate_alpha.py appends repo root to sys.path before the import. `cd /tmp && python <abs>/tools/estimate_alpha.py --help` exits 0.
- C2: `priors.prior_map_path(dirpath, rPU, split=None)`, always with `_{split}` infix when split given; used by model.extract_similarities (split=None keeps legacy name) and estimate_alpha. March files untouched, now unreachable. Tests: test_priors (infix, legacy name). README filename updated.
- C3: gen_jobs.py uses `x=$((x + 1))`; jobs regenerated. Verified `set -euo pipefail` arithmetic does not abort (exit 0, count 2); test runs the generated increments under set -e in a subprocess. submit_all.sh not executed.
- C4: aggregate() raises ValueError on non-cited group size != 3 unless --allow-partial (then warns). Markdown shows (n=K) on every non-cited cell; LaTeX annotates only partial cells. Empty/missing metrics dir raises. UTF-8 encoding added. Tests: fewer/more than 3 seeds, allow-partial both renderers, cited has no n, empty/missing dir.
- I5: fs_pt_ft must be cite:-sourced (ValueError). README reworded to call a local Han run a sanity check. Test added.
- I6: mixed non-null alpha -> ValueError; differing commit -> warning. Tests added.
- I7: README seed-0 paragraph says reuse is unverified pending `6 of 6 matched` and that seed-0 records are not in the repo; no backfill script invented. Procedure steps 1-3 now runnable.
- Minor: PRIOR_DIR and STABLE_TAU_WINDOW constant in estimate_alpha; check_equivalence.sh cds to repo root; job template does `cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../.."}"` (under sbatch $0 is a spooled copy, so dirname alone would be wrong); --epochs type=int; test_committed_jobs_match_the_generator added.

Not done / caveats: nothing was run on GPU; the estimate_prior.sh chain end to end was not executed (needs GPU extraction). The 2-seed groups for supervised_ft/pu_zero/pu_exact will make aggregate_table error until seed-0 records exist (intended).
