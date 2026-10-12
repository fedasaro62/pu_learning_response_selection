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

Two independent flags decide what a run is. `--post-train` picks the checkpoint (`positive_only`
from step 1, `original` for Han et al.'s released BERT-FP, `zero_shot` for the plain encoder), and
`--objective` picks the loss (`pu` or `supervised`). For `pu`, `--prior` sets `α`: `zero` treats
every unlabeled pair as a negative, `exact` computes `α*` from the label counts, and `estimated`
reads our `α̂` from step 3.

`scripts/finetune.sh` takes `task objective prior seed rPU`, always loads our post-trained
checkpoint, and reports R@1, R@2 and R@5 on the test set:

```bash
scripts/finetune.sh ubuntu supervised exact     0 0.8   # PU_PT + FS_FT
scripts/finetune.sh ubuntu pu         zero      0 0.8   # PU_PT+FT, alpha = 0
scripts/finetune.sh ubuntu pu         exact     0 0.8   # PU_PT+FT, alpha = alpha*
scripts/finetune.sh ubuntu pu         estimated 0 0.8   # PU_PT+FT, alpha = alpha-hat
```

The `FS_PT+FT` row is cited from Han et al. rather than re-run. To reproduce it as a sanity check,
call the entrypoint directly (`--rPU` is not needed, since neither the checkpoint nor the loss uses it):

```bash
python FineTuning/Response_selection.py --task ubuntu --post-train original --objective supervised --seed 0 --is_training
```

### 3. Our class prior estimator

```bash
scripts/estimate_prior.sh ubuntu 0.8
```

Extracts the mean-pooled context and response embeddings from the post-trained encoder, caches their
cosine similarities, then sweeps `τ` over `{0.50, ..., 0.90}`, computes the tail ratio `α̂(τ)` and
picks `τ*` as the centre of the lowest-variance window of 5 consecutive grid points. The estimate is
written to `resources/prior/alpha_hat_<task>_rPU<r>_<split>.json`, which `--prior estimated` reads,
so run this before any `estimated` fine-tuning. The train-split estimate is the one reported below.

| Ubuntu (train) | E-Commerce (train) |
| --- | --- |
| ![](docs/alpha_vs_tau_ubuntu_train.png) | ![](docs/alpha_vs_tau_e_commerce_train.png) |

`α̂(τ)` falls from near 1, flattens onto a plateau around the true `α*` (red), then goes noisy once
the tail empties. The selected `τ*` (black) lands inside that plateau on both corpora.

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
That separation is what makes the tail-ratio estimator work.

## Reported results

Response selection under `β = 10%`, averaged over three fine-tuning seeds:

| Setting | Ubuntu R@1 | Ubuntu R@2 | Ubuntu R@5 | E-Commerce R@1 | E-Commerce R@2 | E-Commerce R@5 |
| --- | --- | --- | --- | --- | --- | --- |
| Fully supervised (Han et al.) | 0.911 | 0.962 | 0.994 | 0.870 | 0.956 | 0.993 |
| PU post-training + supervised fine-tuning | 0.877 | 0.941 | 0.988 | 0.780 | 0.907 | 0.986 |
| Fully PU, `α = 0` | 0.813 | 0.909 | 0.975 | 0.544 | 0.765 | 0.954 |
| Fully PU, `α = α*` | 0.824 | 0.918 | 0.981 | 0.667 | 0.852 | 0.972 |
| **Fully PU, `α = α̂`** | **0.816** | **0.916** | **0.981** | **0.646** | **0.826** | **0.964** |

Class prior estimates against `α* = 0.444` on both corpora:

| Estimator | Ubuntu α̂ | Ubuntu abs. dev. | E-Commerce α̂ | E-Commerce abs. dev. |
| --- | --- | --- | --- | --- |
| EN | 0.547 | 0.103 | 0.423 | 0.021 |
| Scott | 0.236 | 0.208 | 0.507 | 0.063 |
| BBE | 0.356 | 0.088 | 0.471 | 0.027 |
| Ours | 0.435 | 0.009 | 0.459 | 0.015 |

`tools/aggregate_table.py` rebuilds the first table from the per-run metrics; `scripts/gen_jobs.py`
generates the Slurm arrays for all 24 runs.

## Attribution

Built on [BERT-FP](https://github.com/hanjanghoon/BERT_FP) (Han et al., NAACL 2021). The
mixture-proportion-estimation baselines come from
[acmi-lab/PU_learning](https://github.com/acmi-lab/PU_learning) (Garg et al., NeurIPS 2021), which
is Apache-2.0 licensed. See `NOTICE` for details, and `LICENSE` for the license text.
