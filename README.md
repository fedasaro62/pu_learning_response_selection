# Positive-Unlabeled Learning for Response Selection

Reference implementation for the paper *Positive-Unlabeled Learning for Response Selection*.

Response selection is cast as a Positive-Unlabeled (PU) problem: only a fraction of the matching
context-response pairs is labeled, and everything else, including all true negatives, is unlabeled.
The repository covers the full PU pipeline on top of BERT-FP, plus a lightweight class prior
estimator that reads the cosine similarities of the post-trained encoder and needs no labeled
negatives and no second model.

![BERT-FP architecture](docs/architecture.png)

The context and response are encoded as one sequence; the final `[CLS]` state is scored by a linear
layer. Post-training and fine-tuning both run on this backbone.

## The PU ratio

Throughout the code the flag is `--rPU`, the fraction of positives *hidden* in the unlabeled pool.
The paper reports the labeled fraction `β = N_L / N_D` instead. On the 1:1 training sets of both
corpora the two are related by `β = (1 - rPU) / 2`, so the paper's `β = 10%` setting corresponds to
`--rPU 0.8`: 100K labeled positives and 900K unlabeled pairs out of 1M, and an exact class prior
`α* = 0.444`.

## Setup

Python 3.10 or newer, and one CUDA GPU with at least 40GB for the full runs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Then follow `data/README.md` to download the Ubuntu and E-Commerce corpora and build the pickles.
All commands below are run from the repository root.

## Reproducing the experiments

### 1. PU post-training

```bash
scripts/post_train.sh ubuntu 0.8
scripts/post_train.sh e_commerce 0.8
```

Writes one checkpoint per epoch under `FPT/PT_checkpoint/positive_only/<task>/checkpoint<n>/`. The
later stages read epoch 25 for Ubuntu and epoch 34 for E-Commerce, matching the BERT-FP schedule.

### 2. Fine-tuning

Two flags decide what a fine-tuning run is, and they are independent. `--post-train` picks which
checkpoint is loaded (`positive_only` is ours from step 1, `original` is Han et al.'s released
BERT-FP, `zero_shot` is the plain pretrained encoder). `--objective` picks the loss: `pu` is the PU
loss over the partially labeled targets, `supervised` is plain cross-entropy over the full labels.
The old `--strategy` flag is gone, and so is the old coupling where `--rPU` silently chose both the
checkpoint and the loss. That coupling made PU post-training followed by supervised fine-tuning
unreachable from the command line; it is now just `--post-train positive_only --objective supervised`.

For the `pu` objective, `--prior` sets `α`: `zero` treats every unlabeled pair as a negative, `exact`
uses `α*` computed from the label counts, and `estimated` uses our `α̂` from step 3. The prior is
ignored under `supervised`.

`scripts/finetune.sh` takes `task objective prior seed rPU`, always loads our post-trained
checkpoint, and trains and then evaluates R@1, R@2 and R@5 on the test set. The four table rows that
use it are:

```bash
scripts/finetune.sh ubuntu supervised exact     0 0.8   # PU_PT + FS_FT
scripts/finetune.sh ubuntu pu         zero      0 0.8   # PU_PT+FT, alpha = 0
scripts/finetune.sh ubuntu pu         exact     0 0.8   # PU_PT+FT, alpha = alpha*
scripts/finetune.sh ubuntu pu         estimated 0 0.8   # PU_PT+FT, alpha = alpha-hat
```

The fully supervised baseline of Han et al. is not a `finetune.sh` run, because that script always
loads our checkpoint. Call the entrypoint directly; `--rPU` is optional for this combination since
neither the checkpoint nor the loss needs it:

```bash
python FineTuning/Response_selection.py --task ubuntu --post-train original --objective supervised --seed 0 --is_training
```

It loads the checkpoint from `FPT/PT_checkpoint/original/<task><n>/bert.pt`, and its metrics are
recorded under the `fs_pt_ft` row. That run is a sanity check only: the table's `FS_PT+FT` row is cited
from Han et al., and `tools/aggregate_table.py` refuses any `fs_pt_ft` record that is not `cite:`-sourced. A
`--post-train zero_shot` run is tagged `zero_shot_ft`, which is not a paper row, and
`tools/aggregate_table.py` rejects it as unknown.

### 3. Our class prior estimator

```bash
scripts/estimate_prior.sh ubuntu 0.8
```

This runs two steps per split. First it extracts the mean-pooled context and response embeddings
from the post-trained encoder and caches their cosine similarities as
`prior_map_cosine_rPU0.8_{train,dev}.json`. Then it sweeps `τ` over `{0.50, ..., 0.90}`, computes the tail
ratio `α̂(τ)`, picks `τ*` as the centre of the lowest-variance window of 5 consecutive grid points,
and writes `results/pi_hat_vs_tau_<split>_<task>_rPU0.8.png`.

The estimate printed for the train split is the one reported for our estimator below.

| Ubuntu (dev) | E-Commerce (dev) |
| --- | --- |
| ![](docs/alpha_vs_tau_ubuntu_dev.png) | ![](docs/alpha_vs_tau_e_commerce_dev.png) |

`α̂(τ)` falls from near 1, flattens onto a plateau around the true `α*` (red), then goes noisy once
the tail empties. The selected `τ*` (black) lands inside that plateau on both corpora.

### 3b. Window sensitivity of `τ*`

```bash
uv run tools/ablate_window.py            # --split train --rpu 0.8 --windows 3 5 7 9 11
```

`w = 5` was chosen by inspecting the curves. This tool re-runs `select_stable_tau` from
`tools/estimate_alpha.py` for each window on the persisted `resources/prior/alpha_hat_<task>_rPU0.8_<split>.json`
curves (no GPU, no recomputation of the curve) and writes `resources/window_ablation.{md,tex}`: `τ*`, `α̂(τ*)`
and `|Δ| = |α̂ − α*|` per window and corpus, with the paper's window marked. Read the table as-is; whether
`τ*` moves with `w` is an outcome of the data. Use `--split dev` to compare against the dev curves.

### 4. Baseline estimators

```bash
scripts/baseline_estimators.sh ubuntu 0.8
```

Fine-tunes a positive-versus-unlabeled classifier on top of the post-trained encoder and applies EN,
Scott, BBE and DedPU to its output scores. Estimates are appended to
`PvU/pvu_checkpoint/<task>_rPU0.8_new/{train,dev}_estimate_alpha.csv`, with `α*` recorded in
`true_alpha.csv` for reference.

### 5. Similarity distributions

```bash
scripts/plot_similarities.sh ubuntu 0.8
```

| Zero-shot | Post-trained |
| --- | --- |
| ![](docs/similarity_zero_shot.png) | ![](docs/similarity_post_trained.png) |

Post-training pulls matched and unmatched pairs apart on Ubuntu, from `Δμ = 0.028` to `Δμ = 0.112`.
That separation is what makes the tail-ratio estimator work. Output goes to
`results/density/cosine_similarity/`.

## Three-seed protocol

The table row `α = α̂` (not yet in the Reported results table below) and the seed statistics need three fine-tuning seeds (0, 1 and 2) per cell, 24 runs in
total (four rows, two corpora, three seeds). Order matters in one place:

1. Run `scripts/estimate_prior.sh ubuntu 0.8` and `scripts/estimate_prior.sh e_commerce 0.8` first.
   Besides printing the estimate, each call persists it to
   `resources/prior/alpha_hat_<task>_rPU<r>_<split>.json`, and `--prior estimated` reads that file.
   `jobs/submit_all.sh` handles this ordering itself (see step 2).
2. Generate and submit the jobs. `python scripts/gen_jobs.py` writes four Slurm files plus `jobs/submit_all.sh`:
   `jobs/estimate_prior.job` (array, one element per corpus), `jobs/train_main.job` (array of 18: the supervised,
   zero-prior and exact-prior rows), `jobs/train_estimated.job` (array of 6: the estimated-prior row) and
   `jobs/equivalence.job` (a single diagnostic job). Each array element prints its configuration first, since
   the log name `logs/<group>/<array id>_<index>.out` does not identify the run. `jobs/submit_all.sh` submits
   `train_main.job`, then `train_estimated.job`: directly if both prior files exist, otherwise after
   `estimate_prior.job` with `--dependency=afterok`. It does not submit `equivalence.job`; run
   `sbatch jobs/equivalence.job` yourself. Rerunning the generator overwrites every file under `jobs/`, so edit
   the constants in the generator rather than the generated files. The job files assume a Slurm cluster with A100
   partitions and a virtualenv at `.venv` in the repo root (if yours lives elsewhere, `export PU_VENV=<venv dir>`
   before `sbatch`), so edit the `#SBATCH` header in `scripts/gen_jobs.py` for your setup, or just run the `scripts/finetune.sh` line each job contains.
3. Run `python tools/aggregate_table.py`. It reads the per-run metrics and writes `results/table1.md`
   and `results/table1.tex` with mean ± std over seeds. It rejects unknown rows, mixed `rPU` or `alpha`
   groups, and any run group without exactly three seeds rather than averaging them.

Every reported cell comes from runs produced by this code, at seeds 0, 1 and 2. Nothing is reused from an
earlier campaign. `--allow-partial` renders groups with a different seed count, annotated with it.

### Caveats on what the variance means

- The reported spread is fine-tuning variance only. The seed changes the classifier-head
  initialisation and the data order. Post-training and the PU split are held fixed. The split is drawn
  at dataset-build time with `seed=42` in `Data_processing.py`, and re-drawing it would invalidate
  post-training, which trains on the labeled positives only.
- `α̂` is deterministic given the fixed post-training checkpoint, so it is one number per corpus and
  carries no seed variance. The spread in the `α̂` row comes entirely from fine-tuning.
- `torch.backends.cudnn.benchmark` stays enabled. Runs are seed-controlled but not bitwise
  reproducible, and you should expect small differences across GPUs and driver versions.

## Reported results

Response selection under `β = 10%`, for orientation when re-running:

| Setting | Ubuntu R@1 | Ubuntu R@2 | Ubuntu R@5 | E-Commerce R@1 | E-Commerce R@2 | E-Commerce R@5 |
| --- | --- | --- | --- | --- | --- | --- |
| Fully supervised (Han et al.) | 0.911 | 0.962 | 0.994 | 0.870 | 0.956 | 0.993 |
| PU post-training + supervised fine-tuning | 0.877 | 0.941 | 0.988 | 0.775 | 0.913 | 0.987 |
| Fully PU, `α = 0` | 0.813 | 0.909 | 0.975 | 0.544 | 0.765 | 0.954 |
| Fully PU, `α = α*` | 0.824 | 0.918 | 0.981 | 0.667 | 0.852 | 0.972 |

Class prior estimates against `α* = 0.444` on both corpora:

| Estimator | Ubuntu α̂ | Ubuntu abs. dev. | E-Commerce α̂ | E-Commerce abs. dev. |
| --- | --- | --- | --- | --- |
| EN | 0.547 | 0.103 | 0.423 | 0.021 |
| Scott | 0.236 | 0.208 | 0.507 | 0.063 |
| BBE | 0.356 | 0.088 | 0.471 | 0.027 |
| Ours | 0.435 | 0.009 | 0.459 | 0.015 |

## Attribution

Built on [BERT-FP](https://github.com/hanjanghoon/BERT_FP) (Han et al., NAACL 2021). The
mixture-proportion-estimation baselines come from
[acmi-lab/PU_learning](https://github.com/acmi-lab/PU_learning) (Garg et al., NeurIPS 2021), which
is Apache-2.0 licensed. See `NOTICE` for details, and `LICENSE` for the license text.
