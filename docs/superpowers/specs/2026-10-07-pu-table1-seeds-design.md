# Table 1 revision: estimated prior row and multi-seed variance

Date: 2026-10-07
Status: approved design, pending implementation plan

## Goal

Two additions to Table 1 of *Positive-Unlabeled Learning for Response Selection*:

1. A `PU_PT+FT (α=α̂)` row using the prior produced by the paper's own stability-based
   threshold estimator, rather than the oracle `α*`. The table currently shows only `α=0`
   and `α=α*`, so it never demonstrates that the estimator is usable for training — only
   that it is accurate against a held-out truth.
2. Mean ± standard deviation over three fine-tuning seeds for every locally-run row, so
   the reported gaps carry a variance estimate.

`FS_PT+FT` stays as the cited Han et al. constant with no standard deviation. Table 2
(class prior estimates) is out of scope and unchanged.

## Blocking defect found during design

The `PU_PT + FS_FT` row (Ubuntu 0.877/0.941/0.988, E-Commerce 0.775/0.913/0.987) is not
reachable from any flag combination in the released code. Its run logs
(`innonext-BERT_FP/logs/statistical_prior/*_fully_supervised.out`) show an argv identical
to the `α=α*` run — same `--rPU 0.8 --strategy statistical_prior`, printing the same
`Estimated prior: 0.4444` — yet different results. It was produced by hand-editing
`statistical_prior.py` to call `compute_fully_supervised_loss`.

The cause is that `--rPU` controls two unrelated things at once: which post-training
checkpoint loads, and whether the PU loss or plain cross-entropy is used
(`statistical_prior.py:195-198` and `fit()`). "PU post-training + supervised fine-tuning"
needs the first without the second, and no flag expresses that.

Decoupling these is therefore a prerequisite for the revision, not optional cleanup.

## Decisions taken

| Question | Decision |
|---|---|
| What varies across seeds | Fine-tuning only. Post-training checkpoint and the PU split stay fixed. |
| Which rows get mean ± std | The four locally-run rows. `FS_PT+FT` stays a cited constant. |
| Seed 0 | Reuse the existing runs; run seeds 1 and 2 fresh. The new α̂ row runs all three. |
| How α̂ reaches the fine-tuner | The estimator persists it to a JSON file; the trainer loads that file. No CLI override. |
| Execution | Produce code and Slurm job files. The user submits them. |
| Repo | This repo (`pu_learning_response_selection`), with data and checkpoints symlinked in. |

### Consequences to state in the paper

- Reported variance is fine-tuning variance only (initialisation of the classifier head
  and data order). It does not cover post-training or the draw of which positives are
  labeled. The PU split is fixed at dataset-build time with `seed=42`
  (`Data_processing.py:37`), and re-drawing it would invalidate post-training, which
  trains on labeled positives only.
- α̂ is deterministic given the fixed post-training checkpoint, so it is a single number
  per corpus and carries no seed variance.
- `cudnn.benchmark` stays enabled, so runs are seed-controlled but not bitwise
  reproducible across hardware.

## Module layout

`statistical_prior.py` (376 lines), `pos_neg_finetune.py` (358) and `fixed_prior.py` (422)
are near-identical copies differing only in how the unlabeled term is weighted. That
duplication is what allowed the `PU_PT+FS_FT` hand-edit to become unreproducible. They
collapse into:

```
FineTuning/BERT_finetuning/
  data.py      BERTDataset, InputFeatures, _truncate_seq_pair
  model.py     NeuralNetwork: build, fit, evaluate, predict, extract_similarities, seed_everything
  losses.py    pu_loss(logits, pu_labels, alpha), supervised_loss(logits, labels)
  priors.py    resolve_prior(...) -> float | None
```

All three old modules are deleted. `estimate_prior_map.py` currently imports
`NeuralNetwork` and `BERTDataset` from `fixed_prior.py`; its embedding extraction becomes
`model.extract_similarities()` and the imports move to `model.py` / `data.py`. Its external
behaviour and output paths are unchanged.

The per-sample prior-map variant inside `fixed_prior.py` (a prior per example rather than a
scalar) is unused by the paper and is dropped.

Target Python is 3.10 (the cluster venv is 3.10.12), so `X | None` hints are available but
3.12-only syntax is not.

## Flags

On `FineTuning/Response_selection.py`:

```
--task {ubuntu, e_commerce, douban}
--rPU FLOAT|None                                   selects the y_pu_<rPU> column and names the checkpoint
--post-train {positive_only, original, zero_shot}  selects the checkpoint   (default positive_only)
--post-train-ckpt INT                              default per task: ubuntu 25, e_commerce 34, douban 27
--objective {pu, supervised}                       selects the loss         (default pu)
--prior {exact, zero, estimated}                   read only when --objective pu
--seed INT                                         default 0
```

Checkpoint resolution by `--post-train`:

- `positive_only` → `FPT/PT_checkpoint/positive_only/<task>/checkpoint<N>/bert_rPU<rPU>.pt`
- `original` → `FPT/PT_checkpoint/original/<task><N>/bert.pt` (Han et al.'s release)
- `zero_shot` → no checkpoint loaded

`--rPU` no longer decides which loss runs; it still selects the PU label column and appears
in the post-trained checkpoint's filename (`bert_rPU<rPU>.pt`), which is why
`--objective supervised` is still passed `--rPU 0.8`.

The old `--strategy` flag is removed. Table rows map to flags as:

| Table 1 row | Flags |
|---|---|
| `PU_PT + FS_FT` | `--objective supervised --post-train positive_only --rPU 0.8` |
| `PU_PT+FT (α=0)` | `--objective pu --prior zero --rPU 0.8` |
| `PU_PT+FT (α=α*)` | `--objective pu --prior exact --rPU 0.8` |
| `PU_PT+FT (α=α̂)` | `--objective pu --prior estimated --rPU 0.8` |

`--objective supervised` trains on the ground-truth `y` column with full cross-entropy;
`--objective pu` trains on `y_pu_<rPU>` with the PU loss weighted by the resolved α.

## Prior resolution and persistence

`tools/estimate_alpha.py` currently only prints its result. It gains a write to
`resources/prior/alpha_hat_<task>_rPU<rPU>_<split>.json`:

```json
{
  "task": "ubuntu", "rPU": 0.8, "split": "train",
  "tau_star": 0.80, "alpha_hat": 0.4351, "alpha_exact": 0.4444,
  "window": 5, "score_transform": "cosine",
  "taus": [0.50, 0.51, "..."], "alpha_of_tau": [1.0, 1.0, "..."]
}
```

`resources/` is used rather than `results/` because the latter is gitignored, and this file
is a training input that must be tracked for the pipeline to be reproducible from a clone.
Storing the full `(taus, alpha_of_tau)` curve also lets Figure 3 be regenerated without the
30MB prior map.

`priors.resolve_prior`:

- `zero` → `0.0`
- `exact` → `(n_pos - n_labeled) / (n_total - n_labeled)`, the existing formula
- `estimated` → reads the train-split JSON above and returns `alpha_hat`; if the file is
  missing, raises with the exact command needed to produce it
- `--objective supervised` → `None`, and no prior is computed

## Seeding and output paths

`model.seed_everything(seed)` sets `random`, `numpy`, `torch` and `torch.cuda`, and is
called once from the entrypoint before the model is constructed. The module-level
`torch.manual_seed(0)` statements are removed.

`torch.backends.cudnn.benchmark = True` is deliberately retained. Switching to
deterministic kernels would change the seed-0 trajectory and break comparability with the
reused seed-0 runs, as well as slowing training.

### RNG ordering invariant

The `Batch[0]` equivalence check below only holds if the refactor preserves the *order* of
RNG consumption, not merely the seed value. Two draws determine `Batch[0]`: the
classifier head initialisation inside `NeuralNetwork.__init__`, and the first `RandomSampler`
permutation. The invariant is therefore:

> `seed_everything(seed)` is called at the same point in the RNG sequence that
> `torch.manual_seed(0)` occupies today — that is, before `NeuralNetwork` is constructed,
> with nothing consuming torch RNG in between.

Today that statement runs at strategy-module import time, before the pickle load, and
nothing between it and model construction touches torch RNG, so calling it from the
entrypoint is equivalent. Two supporting facts, both verified against the current source:

- Neither `statistical_prior.py` nor `pos_neg_finetune.py` contains any use of `random.` or
  `np.random`, so newly seeding those two generators cannot shift `Batch[0]`.
- Their `fit()` bodies are identical apart from a commented-out prior-estimation block, and
  `estimate_prior(train['y'])` is a pure count over Python lists that consumes no RNG. The
  unified `fit()` therefore reproduces both call sequences exactly.

If a `Batch[0]` check fails, an added or removed RNG consumer between seeding and the first
forward pass is the first thing to suspect.

```
FineTuning/FT_checkpoint/<task>/<row>/seed<k>/bert_rPU0.8.pt
                                             /scorefile_rPU0.8.txt
resources/metrics/<task>_<row>_rPU0.8_seed<k>.json
```

with `row` one of `supervised_ft`, `pu_zero`, `pu_exact`, `pu_estimated`. These names map
1:1 onto table rows. Existing `statistical_prior/` and `pos_neg_finetune/` checkpoint
directories are left untouched.

Metrics are written to `resources/` rather than beside the checkpoints because
`FineTuning/FT_checkpoint/` is gitignored, and the table must be regenerable from a clone.
Each file holds:

```json
{
  "task": "ubuntu", "row": "pu_exact", "rPU": 0.8, "seed": 0, "alpha": 0.4444,
  "MAP": 0.8919, "MRR": 0.8919, "P@1": 0.82434,
  "R10@1": 0.82434, "R10@2": 0.91782, "R10@5": 0.98096,
  "source": "run", "commit": "<sha>", "wall_minutes": 874.4
}
```

The six reused seed-0 runs (three rows on two corpora) get the same file backfilled from
the logs, with `"source": "log:innonext-BERT_FP/logs/<dir>/<file>.out"`, so their
provenance is recorded in the artifact rather than only in the commit message. For those,
`commit` is `null` and `MAP`/`MRR`/`P@1` are filled where the log records them — the logs
do carry all six values, but the schema permits `null` so a backfill is never blocked by a
missing field.

Disk cost is negligible: 18 runs at roughly 420MB per checkpoint is about 7.6GB, against
12TB free on `/nfs` and 89GB currently used by the project.

## Aggregation

`tools/aggregate_table.py` reads `resources/metrics/*.json`, computes mean and sample
standard deviation (`ddof=1`) of R10@1/2/5 across seeds, and writes `results/table1.md`
and a booktabs `results/table1.tex` matching the paper's current column layout. A cell
backed by fewer than three seeds produces a warning rather than a silently averaged value.

`FS_PT+FT` is emitted with no standard deviation, read from
`resources/metrics/cited_fs_pt_ft.json`, which carries the published Han et al. numbers and
a `"source": "cite:han2021"` field. It is a tracked input, not a constant buried in the
aggregation script.

On significance: with three seeds the output is mean ± std, and no p-values are claimed.
A t-test at n=3 would not survive review. If an explicit "α̂ beats α=0" claim is wanted
later, the options are to show the gap exceeds the summed standard deviations, or to raise
to five seeds (+12 runs, roughly 155 GPU-hours).

## Run matrix

18 new runs, at roughly 13 GPU-hours each (observed: Ubuntu 874 min, E-Commerce 705 min),
so about 234 GPU-hours total.

| Row | Ubuntu seeds | E-Commerce seeds | New runs |
|---|---|---|---|
| `supervised_ft` | 1, 2 | 1, 2 | 4 |
| `pu_zero` | 1, 2 | 1, 2 | 4 |
| `pu_exact` | 1, 2 | 1, 2 | 4 |
| `pu_estimated` | 0, 1, 2 | 0, 1, 2 | 6 |

`scripts/finetune.sh` gains seed, objective and prior passthrough. `scripts/gen_jobs.py`
emits `jobs/<row>/<task>_seed<k>.job` from the existing template (A100 partition,
`3g.40gb:1`, 16 CPUs, 64G, 48h) plus a `jobs/submit_all.sh` listing the sbatch calls. The
jobs are handed to the user to submit; this work does not run them.

The α̂ estimates must be produced before any `pu_estimated` job starts. The prior maps
already exist on disk, so this is a seconds-long `tools/estimate_alpha.py` run per corpus,
not a re-extraction.

## Equivalence check before launching

Reusing seed 0 requires that the refactored code path behaves like the code that produced
those runs. Every old log records its early per-batch losses, which pin down model
initialisation, data order and the resolved α simultaneously:

| Run | Batch[0] loss | Source log |
|---|---|---|
| Ubuntu `pu_exact` | 1.188617 | `statistical_prior/ubuntu_fine_tune_rPU0.8.out` |
| Ubuntu `pu_zero` | 1.269248 | `pos_neg_finetune/ubuntu_fine_tune_rPU0.8.out` |
| Ubuntu `supervised_ft` | 0.551310 | `statistical_prior/ubuntu_fine_tune_rPU0.8_fully_supervised.out` |
| E-Commerce `pu_exact` | 1.622780 | `statistical_prior/e_commerce_fine_tune_rPU0.8.out` |
| E-Commerce `pu_zero` | 1.487587 | `pos_neg_finetune/e_commerce_fine_tune_rPU0.8.out` |
| E-Commerce `supervised_ft` | 0.705367 | `statistical_prior/e_commerce_fine_tune_rPU0.8_fully_supervised.out` |

For each of the six reused runs, the refactored code runs for a few hundred batches at
seed 0 and its `Batch[0]` loss is compared against the logged value. This costs about 20
minutes per run, roughly two hours in total, instead of 78 hours of full re-runs. Later batches will drift because kernels are nondeterministic, so only
`Batch[0]` is a tight check and batches 100–300 are a loose sanity check.

**If a `Batch[0]` check disagrees, implementation stops and the discrepancy is reported to
the user.** It is not resolved by quietly re-running the row.

## Testing

pytest, fixtures in `conftest.py`, no test classes. `pytest` is not currently in
`requirements.txt` and must be added.

- `tests/test_priors.py` — the exact-prior formula against known counts; the estimated-prior
  loader round-trip; the error raised when the estimate file is absent
- `tests/test_losses.py` — the PU loss at α=0 reduces exactly to the old `pos_neg_finetune`
  loss (its unlabeled term becomes `ℓ⁻(U)`), and at α=1 its unlabeled term becomes `ℓ⁺(U)`;
  plus the supervised path. The α=0 case is the algebraic justification for deleting
  `pos_neg_finetune.py`, so it is the load-bearing test of the refactor.
- `tests/test_aggregate.py` — mean and std from synthetic metric files; the fewer-than-three-seeds
  warning; LaTeX output shape

The GPU equivalence check is a manual script under `scripts/`, not a pytest test, since it
needs a GPU and the real checkpoints.

## Repository wiring

This repo has no data or checkpoints. It gains symlinks matching the live copy:

```
data/ubuntu_data/dataset_1M.pkl      -> the built pickle
data/e_commerce_data/dataset_1M.pkl  -> the built pickle
FPT/PT_checkpoint                    -> /nfs/projects/innonext/models/BERT_FP_PU/PT_checkpoint
FineTuning/FT_checkpoint             -> /nfs/projects/innonext/models/BERT_FP_PU/FT_checkpoint
```

All four are already gitignored.

## Documentation

`README.md` needs updating for the new flags (the `--strategy` examples become
`--objective`/`--prior`), the seed argument, the α̂ row, and a reproduction section
covering the three-seed protocol. The "Reported results" tables are regenerated from
`results/table1.md` once runs complete.

## Out of scope

- Table 2 and the baseline estimators in `PvU/`
- Re-running post-training, or varying the PU split across seeds
- The `douban` corpus, which the paper does not report
- Any change to the `rPU=0.9` setting
