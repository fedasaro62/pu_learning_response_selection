# Table 1 Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `PU_PT+FT (α=α̂)` row to Table 1 and report mean ± std over three fine-tuning seeds for every locally-run row.

**Architecture:** Three near-identical fine-tuning modules collapse into one `NeuralNetwork` parameterised by a scalar prior, with `--objective` selecting the loss and `--post-train` selecting the checkpoint — decoupling two concerns that `--rPU` currently conflates. A prior resolver supplies α from one of three sources, one of which is a JSON file the estimator now persists. Per-run metrics are written as tracked JSON and aggregated into the paper table.

**Tech Stack:** Python 3.10.12, PyTorch 2.5.1+cu121, transformers 4.57.5, numpy 2.2.6, pytest, Slurm.

**Spec:** `docs/superpowers/specs/2026-10-07-pu-table1-seeds-design.md`

## Global Constraints

- Python 3.10.12 — `X | None` hints available, 3.12-only syntax is not. Use `from __future__ import annotations` in new modules.
- PEP 8, line length 119, 4 spaces. Type hints on all functions. Comments start lowercase. No docstring at the top of a file. No docstrings on single-use inner helpers.
- pytest, fixtures in `tests/conftest.py`, no test classes. Run with `-s`.
- Prefer dict dispatch or inline conditionals over `if`/`elif` chains.
- `torch.backends.cudnn.benchmark = True` must stay enabled — changing it breaks comparability with the reused seed-0 runs.
- Seeds are `0, 1, 2`. `rPU` is `0.8` throughout. Corpora are `ubuntu` and `e_commerce` only.
- Row tags are exactly `supervised_ft`, `pu_zero`, `pu_exact`, `pu_estimated`, plus `fs_pt_ft` for the cited row.
- **RNG ordering invariant:** `seed_everything(seed)` must run before `NeuralNetwork` is constructed, with nothing consuming torch RNG in between. Violating this breaks every equivalence check in Task 9.

## Review Focus

Five failure modes the spec implies that no task's happy path exercises. Each has a test assigned to the task owning the code.

1. **`--objective supervised` with `--rPU` omitted** builds the path `bert_rPUNone.pt` and dies inside `torch.load` with an unrelated message. Expect an argparse-level error naming the flag. → Task 5.
2. **`--prior estimated` resolving against a dev-split estimate** would train on the wrong α with no warning. Expect the loader to reject any file whose `split` is not `train`. → Task 2.
3. **Duplicate metrics files for one seed** (a re-run writing under a second name) make the aggregator average one run twice and understate std. Expect a hard error naming the duplicated seed. → Task 7.
4. **A killed job leaving a truncated metrics JSON** makes the aggregator raise `JSONDecodeError` naming no file. Expect an error naming the bad path. → Task 7.
5. **numpy seeding changes `Metrics` tie-breaking**, because `evaluation_one_session` calls `np.random.shuffle` before sorting. Evaluation differences could be misattributed to the model. Expect metrics to be reproducible for a fixed seed and scorefile. → Task 4.

---

### Task 1: pytest scaffolding and `losses.py`

Extracts the one piece of real logic that differs between the three strategy modules. The α=0 test is the algebraic justification for deleting `pos_neg_finetune.py`, so it is the load-bearing test of the whole refactor.

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Create: `FineTuning/BERT_finetuning/__init__.py`, `tools/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_losses.py`
- Create: `FineTuning/BERT_finetuning/losses.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `supervised_loss(logits: Tensor, labels: Tensor) -> Tensor`, `pu_loss(logits: Tensor, pu_labels: Tensor, alpha: float) -> Tensor`. PU labels use the existing encoding: `1` labeled positive, `-1` unlabeled.

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
pytest>=7.0.0
```

`torch` is intentionally absent — the README installs it separately from the CUDA index.

- [ ] **Step 2: Make the project importable from tests**

`FineTuning/BERT_finetuning/` and `tools/` have no `__init__.py`, and there is no pytest
config, so `from FineTuning.BERT_finetuning.losses import ...` will not resolve. Every test
in this plan depends on fixing that first.

```bash
touch FineTuning/BERT_finetuning/__init__.py tools/__init__.py
```

`pytest.ini` (`pythonpath` requires pytest ≥ 7.0, which Step 1 pins):

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 3: Verify the import path resolves**

Run: `python -m pytest --collect-only 2>&1 | tail -3`
Expected: `no tests ran` rather than an import or config error.

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import pytest
import torch


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor([0.9, 0.8, 0.3, 0.1, 0.6])


@pytest.fixture
def pu_labels() -> torch.Tensor:
    # two labeled positives, three unlabeled
    return torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])
```

- [ ] **Step 5: Write the failing tests**

`tests/test_losses.py`:

```python
import torch
import torch.nn as nn

from FineTuning.BERT_finetuning.losses import pu_loss, supervised_loss

bce = nn.BCELoss(reduction="none")


def test_pu_loss_at_alpha_zero_matches_pos_neg_objective(logits, pu_labels):
    # the old pos_neg_finetune loss: positives to 1, every unlabeled to 0
    pos, unlab = logits[pu_labels == 1], logits[pu_labels == -1]
    expected = bce(pos, torch.ones_like(pos)).mean() + bce(unlab, torch.zeros_like(unlab)).mean()

    assert torch.allclose(pu_loss(logits, pu_labels, 0.0), expected)


def test_pu_loss_at_alpha_one_treats_unlabeled_as_positive(logits, pu_labels):
    pos, unlab = logits[pu_labels == 1], logits[pu_labels == -1]
    expected = bce(pos, torch.ones_like(pos)).mean() + bce(unlab, torch.ones_like(unlab)).mean()

    assert torch.allclose(pu_loss(logits, pu_labels, 1.0), expected)


def test_pu_loss_interpolates_between_the_two_extremes(logits, pu_labels):
    at_zero = pu_loss(logits, pu_labels, 0.0)
    at_one = pu_loss(logits, pu_labels, 1.0)
    at_half = pu_loss(logits, pu_labels, 0.5)

    assert torch.allclose(at_half, 0.5 * (at_zero + at_one))


def test_pu_loss_handles_batch_with_no_unlabeled(logits):
    labels = torch.ones_like(logits)
    expected = bce(logits, torch.ones_like(logits)).mean()

    assert torch.allclose(pu_loss(logits, labels, 0.4), expected)


def test_pu_loss_handles_batch_with_no_positives(logits):
    labels = torch.full_like(logits, -1.0)
    expected = (0.4 * bce(logits, torch.ones_like(logits))
                + 0.6 * bce(logits, torch.zeros_like(logits))).mean()

    assert torch.allclose(pu_loss(logits, labels, 0.4), expected)


def test_supervised_loss_is_mean_bce(logits):
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])

    assert torch.allclose(supervised_loss(logits, labels), bce(logits, labels).mean())
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest -s tests/test_losses.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'FineTuning.BERT_finetuning.losses'`

- [ ] **Step 7: Implement `losses.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn

_bce = nn.BCELoss(reduction="none")


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return _bce(logits, labels).mean()


def pu_loss(logits: torch.Tensor, pu_labels: torch.Tensor, alpha: float) -> torch.Tensor:
    positive_mask = pu_labels == 1
    unlabeled_mask = pu_labels == -1

    loss = logits.new_zeros(())

    if positive_mask.any():
        pos_logits = logits[positive_mask]
        loss = loss + _bce(pos_logits, torch.ones_like(pos_logits)).mean()

    if unlabeled_mask.any():
        unlab_logits = logits[unlabeled_mask]
        loss_y1 = _bce(unlab_logits, torch.ones_like(unlab_logits))
        loss_y0 = _bce(unlab_logits, torch.zeros_like(unlab_logits))
        loss = loss + (alpha * loss_y1 + (1 - alpha) * loss_y0).mean()

    return loss
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest -s tests/test_losses.py`
Expected: 6 passed

- [ ] **Step 9: Commit**

```bash
git add requirements.txt pytest.ini tests/conftest.py tests/test_losses.py \
        FineTuning/BERT_finetuning/__init__.py tools/__init__.py \
        FineTuning/BERT_finetuning/losses.py
git commit -m "Add unified PU and supervised losses with pytest scaffolding"
```

---

### Task 2: `priors.py`

Resolves α from one of three sources. Owns Review Focus item 2.

**Files:**
- Create: `FineTuning/BERT_finetuning/priors.py`
- Create: `tests/test_priors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `exact_prior(y: Sequence[int], rPU: float) -> float`, `estimated_prior_path(task: str, rPU: float, split: str = "train") -> Path`, `load_estimated_prior(task: str, rPU: float) -> float`, `resolve_prior(objective: str, prior: str, task: str, rPU: float, y: Sequence[int]) -> float | None`. Task 6 writes the file that `load_estimated_prior` reads; Task 5 calls `resolve_prior`.

- [ ] **Step 1: Write the failing tests**

`tests/test_priors.py`:

```python
import json

import pytest

from FineTuning.BERT_finetuning import priors


def test_exact_prior_matches_the_published_value():
    # 1M pairs at 1:1, rPU=0.8 -> 100K labeled positives, alpha* = 0.444
    y = [1] * 500_000 + [0] * 500_000

    assert priors.exact_prior(y, 0.8) == pytest.approx(0.4444, abs=1e-4)


def test_exact_prior_is_one_when_nothing_is_hidden():
    y = [1] * 10 + [0] * 0

    assert priors.exact_prior(y, 1.0) == pytest.approx(1.0)


def test_resolve_prior_zero_ignores_the_labels():
    assert priors.resolve_prior("pu", "zero", "ubuntu", 0.8, y=[1, 0]) == 0.0


def test_resolve_prior_supervised_returns_none():
    assert priors.resolve_prior("supervised", "exact", "ubuntu", 0.8, y=[1, 0]) is None


def test_load_estimated_prior_reads_alpha_hat(tmp_path, monkeypatch):
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)
    path = tmp_path / "alpha_hat_ubuntu_rPU0.8_train.json"
    path.write_text(json.dumps({"alpha_hat": 0.4351, "split": "train"}))

    assert priors.load_estimated_prior("ubuntu", 0.8) == pytest.approx(0.4351)


def test_load_estimated_prior_names_the_command_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=r"scripts/estimate_prior\.sh ubuntu 0\.8"):
        priors.load_estimated_prior("ubuntu", 0.8)


def test_load_estimated_prior_rejects_a_dev_split_estimate(tmp_path, monkeypatch):
    # review focus 2: a dev estimate must never be used to train
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)
    path = tmp_path / "alpha_hat_ubuntu_rPU0.8_train.json"
    path.write_text(json.dumps({"alpha_hat": 0.51, "split": "dev"}))

    with pytest.raises(ValueError, match="split"):
        priors.load_estimated_prior("ubuntu", 0.8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s tests/test_priors.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'FineTuning.BERT_finetuning.priors'`

- [ ] **Step 3: Implement `priors.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

PRIOR_DIR = Path("resources/prior")


def exact_prior(y: Sequence[int], rPU: float) -> float:
    n_pos = int(sum(y))
    n_labeled = round((1 - rPU) * n_pos)
    return (n_pos - n_labeled) / (len(y) - n_labeled)


def estimated_prior_path(task: str, rPU: float, split: str = "train") -> Path:
    return PRIOR_DIR / f"alpha_hat_{task}_rPU{rPU}_{split}.json"


def load_estimated_prior(task: str, rPU: float) -> float:
    path = estimated_prior_path(task, rPU)
    if not path.exists():
        raise FileNotFoundError(
            f"no estimated prior at {path}; produce it with:\n"
            f"    scripts/estimate_prior.sh {task} {rPU}"
        )

    payload = json.loads(path.read_text())
    if payload.get("split") != "train":
        raise ValueError(
            f"{path} holds a '{payload.get('split')}' split estimate; "
            "only the train split may be used to train"
        )
    return float(payload["alpha_hat"])


def resolve_prior(objective: str, prior: str, task: str, rPU: float, y: Sequence[int]) -> float | None:
    if objective == "supervised":
        return None

    sources = {
        "zero": lambda: 0.0,
        "exact": lambda: exact_prior(y, rPU),
        "estimated": lambda: load_estimated_prior(task, rPU),
    }
    return sources[prior]()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s tests/test_priors.py`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add FineTuning/BERT_finetuning/priors.py tests/test_priors.py
git commit -m "Add class prior resolver with train-split guard"
```

---

### Task 3: `data.py`

Lifts the dataset code duplicated across all three strategy modules into one place and repoints the one external consumer that can be moved now.

**Files:**
- Create: `FineTuning/BERT_finetuning/data.py`
- Modify: `tools/plot_similarity_distributions.py:20`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BERTDataset(args, data, tokenizer)`, `InputFeatures`, `_truncate_seq_pair(tokens_a, tokens_b, max_length)`. Task 4's `model.py` imports `BERTDataset` from here.

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:

```python
from FineTuning.BERT_finetuning.data import _truncate_seq_pair


def test_truncate_leaves_short_pairs_alone():
    a, b = [1, 2, 3], [4, 5]

    _truncate_seq_pair(a, b, 10)

    assert (a, b) == ([1, 2, 3], [4, 5])


def test_truncate_drops_from_the_front_of_the_context():
    # context is history, so the oldest tokens go first
    a, b = [1, 2, 3, 4, 5], [6]

    _truncate_seq_pair(a, b, 4)

    assert a == [3, 4, 5] and b == [6]


def test_truncate_drops_from_the_end_of_the_response():
    a, b = [1], [2, 3, 4, 5, 6]

    _truncate_seq_pair(a, b, 4)

    assert a == [1] and b == [2, 3, 4]


def test_truncate_alternates_to_shorten_the_longer_sequence():
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]

    _truncate_seq_pair(a, b, 4)

    assert len(a) + len(b) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s tests/test_data.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'FineTuning.BERT_finetuning.data'`

- [ ] **Step 3: Create `data.py` by moving code verbatim**

Copy `InputFeatures`, `BERTDataset` and `_truncate_seq_pair` from `FineTuning/BERT_finetuning/statistical_prior.py:42-170` into the new file unchanged, with this header:

```python
from __future__ import annotations

import torch
from torch.utils.data import Dataset
```

Do not alter their bodies. `BERTDataset` keeps reading `args.rPU` and `data[f'y_pu_{args.rPU}']` exactly as today — a behaviour change here would invalidate every equivalence check in Task 9.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s tests/test_data.py`
Expected: 4 passed

- [ ] **Step 5: Repoint the similarity-plot import**

In `tools/plot_similarity_distributions.py`, replace line 20:

```python
from BERT_finetuning.fixed_prior import BERTDataset  # noqa: E402
```

with:

```python
from BERT_finetuning.data import BERTDataset  # noqa: E402
```

- [ ] **Step 6: Verify the import resolves**

Run: `python -c "import sys; sys.path.insert(0, 'FineTuning'); from BERT_finetuning.data import BERTDataset; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add FineTuning/BERT_finetuning/data.py tests/test_data.py tools/plot_similarity_distributions.py
git commit -m "Extract shared BERTDataset into data module"
```

---

### Task 4: `model.py` and deletion of the three strategy modules

The core of the refactor. Owns Review Focus item 5.

**Files:**
- Create: `FineTuning/BERT_finetuning/model.py`
- Delete: `FineTuning/BERT_finetuning/statistical_prior.py`, `FineTuning/BERT_finetuning/pos_neg_finetune.py`, `FineTuning/BERT_finetuning/fixed_prior.py`
- Modify: `FineTuning/estimate_prior_map.py:10`, `:52-57`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `data.BERTDataset`, `losses.pu_loss`, `losses.supervised_loss`.
- Produces: `seed_everything(seed: int) -> None`, `post_trained_checkpoint(post_train: str, task: str, ckpt: int, rPU: float | None) -> str | None`, `CHECKPOINTS: dict[str, int]`, `FT_MODEL: dict[str, str]`, and `NeuralNetwork(args)` with `.fit(train, dev)`, `.evaluate(dev, is_test=False)`, `.predict(dev)`, `.load_model(path)`, `.extract_similarities(dataloader, split=None) -> dict`. Task 5 constructs `NeuralNetwork`; `estimate_prior_map.py` calls `extract_similarities`.

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:

```python
import numpy as np
import pytest
import torch

from FineTuning.BERT_finetuning.model import CHECKPOINTS, post_trained_checkpoint, seed_everything


def test_post_trained_checkpoint_for_pu_post_training():
    path = post_trained_checkpoint("positive_only", "ubuntu", 25, 0.8)

    assert path == "./FPT/PT_checkpoint/positive_only/ubuntu/checkpoint25/bert_rPU0.8.pt"


def test_post_trained_checkpoint_for_the_released_han_checkpoint():
    path = post_trained_checkpoint("original", "e_commerce", 34, 0.8)

    assert path == "./FPT/PT_checkpoint/original/e_commerce34/bert.pt"


def test_post_trained_checkpoint_is_none_for_zero_shot():
    assert post_trained_checkpoint("zero_shot", "ubuntu", 25, 0.8) is None


def test_checkpoint_defaults_match_the_bert_fp_schedule():
    assert CHECKPOINTS == {"ubuntu": 25, "douban": 27, "e_commerce": 34}


def test_seed_everything_makes_torch_draws_repeatable():
    seed_everything(1)
    first = torch.randn(4)
    seed_everything(1)

    assert torch.equal(first, torch.randn(4))


def test_seed_everything_makes_numpy_draws_repeatable():
    # review focus 5: Metrics.evaluation_one_session shuffles with numpy before sorting,
    # so evaluation is only reproducible if numpy is seeded too
    seed_everything(1)
    first = np.random.rand(4)
    seed_everything(1)

    assert np.array_equal(first, np.random.rand(4))


def test_different_seeds_give_different_draws():
    seed_everything(1)
    first = torch.randn(4)
    seed_everything(2)

    assert not torch.equal(first, torch.randn(4))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s tests/test_model.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'FineTuning.BERT_finetuning.model'`

- [ ] **Step 3: Write `model.py`**

Start from `statistical_prior.py` and apply exactly these changes. Everything not listed stays byte-identical, including the order of statements inside `fit`.

Header — note the module-level `torch.manual_seed(0)` and `torch.cuda.manual_seed_all(0)` are **removed**, while `cudnn.benchmark` stays:

```python
from __future__ import annotations

import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Metrics import Metrics

from .data import BERTDataset
from .losses import pu_loss, supervised_loss

CHECKPOINTS = {'ubuntu': 25, 'douban': 27, 'e_commerce': 34}

FT_MODEL = {
    'ubuntu': 'bert-base-uncased',
    'douban': 'bert-base-chinese',
    'e_commerce': 'bert-base-chinese',
}

MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)}

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def post_trained_checkpoint(post_train: str, task: str, ckpt: int, rPU: float | None) -> str | None:
    paths = {
        'zero_shot': lambda: None,
        'original': lambda: f"./FPT/PT_checkpoint/original/{task}{ckpt}/bert.pt",
        'positive_only': lambda: f"./FPT/PT_checkpoint/positive_only/{task}/checkpoint{ckpt}/bert_rPU{rPU}.pt",
    }
    return paths[post_train]()
```

In `NeuralNetwork.__init__`, replace the `if args.rPU is None: ... else: ...` checkpoint block (`statistical_prior.py:195-198`) with:

```python
        checkpoint = post_trained_checkpoint(args.post_train, args.task, args.post_train_ckpt, args.rPU)
        if checkpoint is None:
            print("zero-shot setting: using pre-trained BERT without loading a post-trained checkpoint")
        else:
            self.bert_model.bert.load_state_dict(
                state_dict=torch.load(checkpoint, map_location=self.device)
            )
```

and replace `self.prior = None` with `self.prior: float | None = args.prior_value`, set by Task 5's entrypoint.

Delete the `estimate_prior` method — Task 2's `priors.exact_prior` replaces it.

In `train_step`, replace the loss branch with:

```python
        loss = (supervised_loss(logits, batch_y) if self.prior is None
                else pu_loss(logits, batch_pu_y, self.prior))
```

In `fit`, delete the prior-estimation block and the `self.loss_func` assignment. **Keep every other statement in its current order** — dataset, sampler, dataloader, optimizer groups, optimizer, epoch loop:

```python
    def fit(self, train, dev):

        if torch.cuda.is_available(): self.cuda()

        dataset = BERTDataset(self.args, train, self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler,
                                num_workers=self.args.num_workers)

        no_decay = ['bias', 'LayerNorm.weight']
        ...
```

Add `extract_similarities`, which is `fixed_prior.py`'s `estimate_prior` method moved across unchanged apart from its name and the directory it writes to. It keeps producing `prior_map_cosine_rPU<rPU>[_<split>].json` at the same path so the existing estimator pipeline and the 30MB maps already on disk stay valid.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s tests/test_model.py`
Expected: 7 passed

- [ ] **Step 5: Repoint `estimate_prior_map.py`**

Replace line 10:

```python
from BERT_finetuning.fixed_prior import NeuralNetwork, BERTDataset
```

with:

```python
from BERT_finetuning.data import BERTDataset
from BERT_finetuning.model import NeuralNetwork
```

Its `main()` builds an `args` namespace by hand, so give it the fields the new `__init__` reads — add after `args.score_file_path = ""`:

```python
    args.objective = "pu"
    args.prior_value = None
```

and change the final call from `model.estimate_prior(dataloader, split=args.split)` to `model.extract_similarities(dataloader, split=args.split)`.

- [ ] **Step 6: Delete the three strategy modules**

```bash
git rm FineTuning/BERT_finetuning/statistical_prior.py \
       FineTuning/BERT_finetuning/pos_neg_finetune.py \
       FineTuning/BERT_finetuning/fixed_prior.py
```

- [ ] **Step 7: Verify nothing still imports them**

Run: `grep -rn "statistical_prior\|pos_neg_finetune\|fixed_prior" --include=*.py --include=*.sh . | grep -v docs/`
Expected: no output. If `scripts/finetune.sh` appears, leave it — Task 10 rewrites it.

- [ ] **Step 8: Commit**

```bash
git add -A FineTuning tests/test_model.py
git commit -m "Unify fine-tuning into one prior-parameterised model"
```

---

### Task 5: Entrypoint flags, paths and metrics output

Owns Review Focus item 1.

**Files:**
- Modify: `FineTuning/Response_selection.py` (full rewrite of the argument and dispatch sections)
- Create: `tests/test_run_config.py`

**Interfaces:**
- Consumes: `priors.resolve_prior`, `model.seed_everything`, `model.NeuralNetwork`, `model.CHECKPOINTS`.
- Produces: `row_tag(objective: str, prior: str) -> str`, `run_paths(task, row, seed, rPU) -> tuple[str, str, str]` returning `(save_path, score_file_path, metrics_path)`, and `write_metrics(path, **fields) -> None`. Task 7 reads the metrics files; Task 10 generates the CLI invocations.

- [ ] **Step 1: Write the failing tests**

`tests/test_run_config.py`:

```python
import json

import pytest

from FineTuning.run_config import row_tag, run_paths, validate_args, write_metrics


def test_row_tag_for_each_table_row():
    assert row_tag("supervised", "exact") == "supervised_ft"
    assert row_tag("pu", "zero") == "pu_zero"
    assert row_tag("pu", "exact") == "pu_exact"
    assert row_tag("pu", "estimated") == "pu_estimated"


def test_run_paths_carry_the_seed():
    save, score, metrics = run_paths("ubuntu", "pu_exact", 2, 0.8)

    assert save == "./FineTuning/FT_checkpoint/ubuntu/pu_exact/seed2/bert_rPU0.8.pt"
    assert score == "./FineTuning/FT_checkpoint/ubuntu/pu_exact/seed2/scorefile_rPU0.8.txt"
    assert metrics == "resources/metrics/ubuntu_pu_exact_rPU0.8_seed2.json"


def test_validate_args_rejects_supervised_without_rpu():
    # review focus 1: otherwise the path becomes bert_rPUNone.pt and torch.load dies obscurely
    with pytest.raises(SystemExit):
        validate_args(objective="supervised", post_train="positive_only", rPU=None)


def test_validate_args_allows_zero_shot_without_rpu():
    validate_args(objective="pu", post_train="zero_shot", rPU=None)


def test_write_metrics_round_trips(tmp_path):
    path = tmp_path / "m.json"

    write_metrics(str(path), task="ubuntu", row="pu_exact", rPU=0.8, seed=0, alpha=0.4444,
                  result=(0.89, 0.89, 0.824, 0.824, 0.918, 0.981), source="run",
                  commit="abc123", wall_minutes=874.4)

    payload = json.loads(path.read_text())
    assert payload["R10@1"] == 0.824
    assert payload["R10@5"] == 0.981
    assert payload["commit"] == "abc123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s tests/test_run_config.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'FineTuning.run_config'`

- [ ] **Step 3: Create `FineTuning/run_config.py`**

The entrypoint imports torch and sets a process title at module scope, which makes it untestable. Put the pure logic here instead.

```python
from __future__ import annotations

import json
import os
import sys

ROW_TAGS = {"zero": "pu_zero", "exact": "pu_exact", "estimated": "pu_estimated"}


def row_tag(objective: str, prior: str) -> str:
    return "supervised_ft" if objective == "supervised" else ROW_TAGS[prior]


def run_paths(task: str, row: str, seed: int, rPU: float | None) -> tuple[str, str, str]:
    base = f"./FineTuning/FT_checkpoint/{task}/{row}/seed{seed}"
    os.makedirs(base, exist_ok=True)
    os.makedirs("resources/metrics", exist_ok=True)
    return (
        f"{base}/bert_rPU{rPU}.pt",
        f"{base}/scorefile_rPU{rPU}.txt",
        f"resources/metrics/{task}_{row}_rPU{rPU}_seed{seed}.json",
    )


def validate_args(objective: str, post_train: str, rPU: float | None) -> None:
    if rPU is None and post_train == "positive_only":
        sys.exit(
            "--rPU is required with --post-train positive_only: it names the checkpoint "
            "(bert_rPU<r>.pt) as well as the PU label column"
        )


def write_metrics(path: str, task: str, row: str, rPU: float | None, seed: int,
                  alpha: float | None, result: tuple[float, ...], source: str,
                  commit: str | None, wall_minutes: float | None) -> None:
    keys = ("MAP", "MRR", "P@1", "R10@1", "R10@2", "R10@5")
    payload = {
        "task": task, "row": row, "rPU": rPU, "seed": seed, "alpha": alpha,
        **dict(zip(keys, result)),
        "source": source, "commit": commit, "wall_minutes": wall_minutes,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

`run_paths` creating directories is a side effect in an otherwise pure function. It is deliberate: every caller needs them, and splitting it would mean three call sites remembering to `makedirs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s tests/test_run_config.py`
Expected: 5 passed

- [ ] **Step 5: Rewrite the entrypoint's argument block**

In `FineTuning/Response_selection.py`, replace the `--rPU`, `--post_train` and `--strategy` arguments with:

```python
parser.add_argument("--rPU", default=None, type=float, choices=[0.8, 0.9],
                    help="PU ratio: selects the y_pu_<rPU> column and names the checkpoint.")
parser.add_argument("--post-train", dest="post_train", default="positive_only",
                    choices=["positive_only", "original", "zero_shot"],
                    help="Which post-trained checkpoint to load.")
parser.add_argument("--post-train-ckpt", dest="post_train_ckpt", default=None, type=int,
                    help="Post-training epoch to load. Defaults per task to the BERT-FP schedule.")
parser.add_argument("--objective", default="pu", choices=["pu", "supervised"],
                    help="'pu' uses the PU loss over y_pu; 'supervised' uses cross-entropy over y.")
parser.add_argument("--prior", default="exact", choices=["exact", "zero", "estimated"],
                    help="Source of the class prior. Ignored when --objective is 'supervised'.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument("--max-batches", dest="max_batches", default=None, type=int,
                    help="Stop after N training batches and skip evaluation. For equivalence checks.")
```

- [ ] **Step 6: Rewrite the dispatch block**

Replace the `if args.strategy == ...` import block and the path construction that follows it with:

```python
from BERT_finetuning.model import CHECKPOINTS, NeuralNetwork, seed_everything
from BERT_finetuning.priors import resolve_prior
from run_config import row_tag, run_paths, validate_args, write_metrics

validate_args(args.objective, args.post_train, args.rPU)
args.post_train_ckpt = args.post_train_ckpt or CHECKPOINTS[args.task]
row = row_tag(args.objective, args.prior)
args.save_path, args.score_file_path, args.metrics_path = run_paths(args.task, row, args.seed, args.rPU)
```

In `__main__`, after the pickle load and before `train_model`:

```python
    args.prior_value = resolve_prior(args.objective, args.prior, args.task, args.rPU, train['y'])
    print(f"row={row} objective={args.objective} prior={args.prior} alpha={args.prior_value}")
    seed_everything(args.seed)
```

`seed_everything` is the last call before `NeuralNetwork` is constructed inside `train_model`. Nothing between them touches torch RNG. This is the Global Constraints RNG invariant — do not move it.

In `test_model`, capture the return of `model.evaluate(test, is_test=True)` and write the metrics file:

```python
def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    result = model.evaluate(test, is_test=True)
    write_metrics(args.metrics_path, task=args.task, row=row, rPU=args.rPU, seed=args.seed,
                  alpha=args.prior_value, result=result, source="run",
                  commit=os.popen("git rev-parse --short HEAD").read().strip() or None,
                  wall_minutes=(time.time() - start) / 60)
```

`NeuralNetwork.evaluate` currently returns `None`; add `return result` as its last statement.

- [ ] **Step 7: Wire `--max-batches` through the training loop**

Declaring the flag is not enough — Task 9's equivalence check cannot run without the
behaviour. In `model.py`'s `fit`, inside the batch loop and immediately after
`avg_loss += loss.item()`:

```python
                if self.args.max_batches is not None and i + 1 >= self.args.max_batches:
                    print(f"stopping after {i + 1} batches (--max-batches)")
                    return
```

Returning from `fit` skips the per-epoch `self.evaluate(dev)`, so no checkpoint is written.
In `Response_selection.py`'s `__main__`, guard the test phase so it does not try to load a
checkpoint that was never saved:

```python
    if args.is_training:
        train_model(train, dev)
        if args.max_batches is None:
            test_model(test)
    else:
        test_model(test)
```

`estimate_prior_map.py` builds its own args namespace and never calls `fit`, but
`NeuralNetwork` touches `args.max_batches` only inside `fit`, so it needs no new field.

- [ ] **Step 8: Verify the CLI parses and rejects the bad combination**

Run: `python FineTuning/Response_selection.py --task ubuntu --objective supervised --post-train positive_only 2>&1 | tail -2`
Expected: the `--rPU is required` message, exit status 1.

Run: `python FineTuning/Response_selection.py --help | grep -E "objective|prior|seed"`
Expected: the three new flags listed.

- [ ] **Step 9: Commit**

```bash
git add FineTuning/Response_selection.py FineTuning/run_config.py tests/test_run_config.py
git commit -m "Decouple objective, checkpoint and prior selection behind explicit flags"
```

---

### Task 6: Persist the estimated prior

**Files:**
- Modify: `tools/estimate_alpha.py`
- Create: `tests/test_estimate_alpha.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `resources/prior/alpha_hat_<task>_rPU<rPU>_<split>.json`, read by Task 2's `load_estimated_prior`. The schema is `{task, rPU, split, tau_star, alpha_hat, alpha_exact, window, score_transform, taus, alpha_of_tau}`.

- [ ] **Step 1: Write the failing test**

`tests/test_estimate_alpha.py`:

```python
import json

import numpy as np

from tools.estimate_alpha import select_stable_tau, write_estimate


def test_select_stable_tau_finds_the_plateau():
    taus = np.arange(0.5, 0.91, 0.01)
    # falls steeply, then sits flat at 0.44, then goes noisy
    pi_hats = np.concatenate([
        np.linspace(1.0, 0.45, 20), np.full(11, 0.44), np.linspace(0.44, 0.05, 10),
    ])

    tau, pi = select_stable_tau(taus, pi_hats, window=5)

    assert pi == 0.44
    assert 0.69 < tau < 0.81


def test_write_estimate_round_trips_through_the_loader(tmp_path, monkeypatch):
    from FineTuning.BERT_finetuning import priors
    monkeypatch.setattr(priors, "PRIOR_DIR", tmp_path)

    write_estimate(tmp_path, task="ubuntu", rpu=0.8, split="train", tau_star=0.80,
                   alpha_hat=0.4351, alpha_exact=0.4444, window=5, score_transform="cosine",
                   taus=np.array([0.5, 0.51]), pi_hats=np.array([1.0, 0.99]))

    path = tmp_path / "alpha_hat_ubuntu_rPU0.8_train.json"
    payload = json.loads(path.read_text())
    assert payload["alpha_hat"] == 0.4351
    assert payload["taus"] == [0.5, 0.51]
    assert priors.load_estimated_prior("ubuntu", 0.8) == 0.4351
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s tests/test_estimate_alpha.py`
Expected: FAIL, `ImportError: cannot import name 'write_estimate'`

- [ ] **Step 3: Add `write_estimate` to `tools/estimate_alpha.py`**

```python
def write_estimate(prior_dir: Path, task: str, rpu: float, split: str, tau_star: float,
                   alpha_hat: float, alpha_exact: float, window: int, score_transform: str,
                   taus: np.ndarray, pi_hats: np.ndarray) -> Path:
    prior_dir.mkdir(parents=True, exist_ok=True)
    path = prior_dir / f"alpha_hat_{task}_rPU{rpu}_{split}.json"
    payload = {
        "task": task, "rPU": rpu, "split": split,
        "tau_star": float(tau_star), "alpha_hat": float(alpha_hat),
        "alpha_exact": float(alpha_exact), "window": window,
        "score_transform": score_transform,
        "taus": [round(float(t), 2) for t in taus],
        "alpha_of_tau": [float(p) for p in pi_hats],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
```

Add `from pathlib import Path` to the imports, and call it in `main()` just before the final prints:

```python
    written = write_estimate(Path("resources/prior"), args.task, args.rpu, args.split,
                             stable_tau, stable_pi, alpha, 5, args.score_transform,
                             taus, pi_hats)
    print(f"wrote {written}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s tests/test_estimate_alpha.py`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tools/estimate_alpha.py tests/test_estimate_alpha.py
git commit -m "Persist the estimated class prior for reuse in fine-tuning"
```

---

### Task 7: Table aggregation

Owns Review Focus items 3 and 4.

**Files:**
- Create: `tools/aggregate_table.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: the metrics schema from Task 5.
- Produces: `load_metrics(metrics_dir: Path) -> list[dict]`, `aggregate(records) -> dict[tuple[str, str], dict[str, tuple[float, float | None]]]` keyed by `(task, row)` then metric, valued `(mean, std_or_None)`, `render_markdown(agg) -> str`, `render_latex(agg) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_aggregate.py`:

```python
import json

import pytest

from tools.aggregate_table import aggregate, load_metrics, render_markdown


def write(dirpath, task, row, seed, r1, source="run"):
    payload = {"task": task, "row": row, "rPU": 0.8, "seed": seed, "alpha": 0.44,
               "MAP": 0.9, "MRR": 0.9, "P@1": r1, "R10@1": r1, "R10@2": 0.92,
               "R10@5": 0.98, "source": source, "commit": None, "wall_minutes": None}
    (dirpath / f"{task}_{row}_rPU0.8_seed{seed}.json").write_text(json.dumps(payload))


def test_aggregate_reports_mean_and_sample_std(tmp_path):
    for seed, r1 in enumerate([0.820, 0.824, 0.828]):
        write(tmp_path, "ubuntu", "pu_exact", seed, r1)

    agg = aggregate(load_metrics(tmp_path))
    mean, std = agg[("ubuntu", "pu_exact")]["R10@1"]

    assert mean == pytest.approx(0.824)
    assert std == pytest.approx(0.004, abs=1e-6)


def test_cited_single_seed_row_has_no_std(tmp_path):
    write(tmp_path, "ubuntu", "fs_pt_ft", 0, 0.911, source="cite:han2021")

    agg = aggregate(load_metrics(tmp_path))

    assert agg[("ubuntu", "fs_pt_ft")]["R10@1"] == (pytest.approx(0.911), None)


def test_fewer_than_three_seeds_warns(tmp_path, recwarn):
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.820)
    write(tmp_path, "ubuntu", "pu_exact", 1, 0.824)

    aggregate(load_metrics(tmp_path))

    assert any("2 of 3 seeds" in str(w.message) for w in recwarn)


def test_duplicate_seed_is_an_error(tmp_path):
    # review focus 3: averaging one run twice would understate std
    write(tmp_path, "ubuntu", "pu_exact", 0, 0.820)
    (tmp_path / "ubuntu_pu_exact_rPU0.8_seed0_copy.json").write_text(
        (tmp_path / "ubuntu_pu_exact_rPU0.8_seed0.json").read_text()
    )

    with pytest.raises(ValueError, match="duplicate seed 0"):
        aggregate(load_metrics(tmp_path))


def test_truncated_file_names_itself(tmp_path):
    # review focus 4: a killed job leaves half a file behind
    (tmp_path / "ubuntu_pu_exact_rPU0.8_seed0.json").write_text('{"task": "ubuntu"')

    with pytest.raises(ValueError, match="ubuntu_pu_exact_rPU0.8_seed0.json"):
        load_metrics(tmp_path)


def test_markdown_formats_mean_plus_minus_std(tmp_path):
    for seed, r1 in enumerate([0.820, 0.824, 0.828]):
        write(tmp_path, "ubuntu", "pu_exact", seed, r1)

    table = render_markdown(aggregate(load_metrics(tmp_path)))

    assert "0.824 ± 0.004" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -s tests/test_aggregate.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.aggregate_table'`

- [ ] **Step 3: Implement `tools/aggregate_table.py`**

```python
from __future__ import annotations

import argparse
import json
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

METRICS = ("R10@1", "R10@2", "R10@5")
TASKS = ("ubuntu", "e_commerce")
ROW_ORDER = ("fs_pt_ft", "supervised_ft", "pu_zero", "pu_exact", "pu_estimated")
ROW_LABELS = {
    "fs_pt_ft": r"FS$_{\mathrm{PT+FT}}$ \cite{han2021fine}",
    "supervised_ft": r"PU$_{\mathrm{PT}}$ + FS$_{\mathrm{FT}}$",
    "pu_zero": r"PU$_{\mathrm{PT+FT}}$ ($\alpha$=0)",
    "pu_exact": r"PU$_{\mathrm{PT+FT}}$ ($\alpha$=$\alpha^{\star}$)",
    "pu_estimated": r"PU$_{\mathrm{PT+FT}}$ ($\alpha$=$\hat{\alpha}$)",
}
EXPECTED_SEEDS = 3


def load_metrics(metrics_dir: Path) -> list[dict]:
    records = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} is not valid JSON, likely a killed run: {exc}") from exc
    return records


def aggregate(records: list[dict]) -> dict[tuple[str, str], dict[str, tuple[float, float | None]]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["row"])].append(record)

    out = {}
    for key, group in grouped.items():
        seeds = [r["seed"] for r in group]
        if (duplicate := next((s for s in seeds if seeds.count(s) > 1), None)) is not None:
            raise ValueError(f"duplicate seed {duplicate} for {key[0]}/{key[1]}")

        cited = all(str(r.get("source", "")).startswith("cite:") for r in group)
        if not cited and len(group) < EXPECTED_SEEDS:
            warnings.warn(f"{key[0]}/{key[1]}: {len(group)} of {EXPECTED_SEEDS} seeds present")

        out[key] = {
            metric: (statistics.fmean(values),
                     statistics.stdev(values) if len(values) > 1 else None)
            for metric in METRICS
            if (values := [r[metric] for r in group])
        }
    return out


def _cell(mean: float, std: float | None) -> str:
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def render_markdown(agg) -> str:
    header = "| Setting | " + " | ".join(
        f"{task} {m}" for task in TASKS for m in METRICS) + " |"
    rule = "| --- " * (1 + len(TASKS) * len(METRICS)) + "|"
    lines = [header, rule]
    for row in ROW_ORDER:
        if not any((task, row) in agg for task in TASKS):
            continue
        cells = [_cell(*agg[(task, row)][m]) if (task, row) in agg else "—"
                 for task in TASKS for m in METRICS]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_latex(agg) -> str:
    lines = [r"\begin{tabular}{l" + "c" * (len(TASKS) * len(METRICS)) + "}", r"\toprule"]
    lines.append("Setting & " + " & ".join(
        rf"$R_{{10}}@{m[-1]}$" for _ in TASKS for m in METRICS) + r" \\")
    lines.append(r"\midrule")
    for row in ROW_ORDER:
        if not any((task, row) in agg for task in TASKS):
            continue
        cells = [_cell(*agg[(task, row)][m]).replace("±", r"$\pm$") if (task, row) in agg else "--"
                 for task in TASKS for m in METRICS]
        lines.append(f"{ROW_LABELS[row]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Table 1 across seeds.")
    parser.add_argument("--metrics-dir", default="resources/metrics", type=Path)
    parser.add_argument("--out-dir", default="results", type=Path)
    args = parser.parse_args()

    agg = aggregate(load_metrics(args.metrics_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table1.md").write_text(render_markdown(agg) + "\n")
    (args.out_dir / "table1.tex").write_text(render_latex(agg) + "\n")
    print(render_markdown(agg))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -s tests/test_aggregate.py`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tools/aggregate_table.py tests/test_aggregate.py
git commit -m "Add multi-seed table aggregation with mean and std"
```

---

### Task 8: Backfill the reused seed-0 metrics

Six runs from the previous campaign become seed 0 of the new table, plus the cited Han et al. row. Their provenance is recorded in the artifact, not just in a commit message.

**Files:**
- Create: `resources/metrics/*.json` (7 files)
- Create: `scripts/backfill_seed0.py`

**Interfaces:**
- Consumes: the metrics schema from Task 5.
- Produces: metrics files that Task 7's aggregator reads.

- [ ] **Step 1: Write the backfill script**

`scripts/backfill_seed0.py`. The values below are the final `Evaluation Result` lines from `/nfs/home/dasaro/research/innonext-BERT_FP/logs/`, in `(MAP, MRR, P@1, R10@1, R10@2, R10@5)` order:

```python
from __future__ import annotations

import json
from pathlib import Path

LOGS = "innonext-BERT_FP/logs"

RUNS = [
    ("ubuntu", "pu_exact", 0.4444, (0.891904, 0.891904, 0.82434, 0.82434, 0.91782, 0.98096),
     f"{LOGS}/statistical_prior/ubuntu_fine_tune_rPU0.8.out", 874.4),
    ("ubuntu", "pu_zero", 0.0, (0.0, 0.0, 0.813, 0.813, 0.909, 0.975),
     f"{LOGS}/pos_neg_finetune/ubuntu_fine_tune_rPU0.8.out", None),
    ("ubuntu", "supervised_ft", None, (0.924171, 0.924171, 0.87688, 0.87688, 0.94094, 0.98844),
     f"{LOGS}/statistical_prior/ubuntu_fine_tune_rPU0.8_fully_supervised.out", 870.9),
    ("e_commerce", "pu_exact", 0.4444, (0.715365, 0.715365, 0.667, 0.667, 0.852, 0.972),
     f"{LOGS}/statistical_prior/e_commerce_fine_tune_rPU0.8.out", 705.6),
    ("e_commerce", "pu_zero", 0.0, (0.715365, 0.715365, 0.544, 0.544, 0.765, 0.954),
     f"{LOGS}/pos_neg_finetune/e_commerce_fine_tune_rPU0.8.out", None),
    ("e_commerce", "supervised_ft", None, (0.867958, 0.867958, 0.775, 0.775, 0.913, 0.987),
     f"{LOGS}/statistical_prior/e_commerce_fine_tune_rPU0.8_fully_supervised.out", 704.4),
]

CITED = [
    ("ubuntu", (0.0, 0.0, 0.911, 0.911, 0.962, 0.994)),
    ("e_commerce", (0.0, 0.0, 0.870, 0.870, 0.956, 0.993)),
]

KEYS = ("MAP", "MRR", "P@1", "R10@1", "R10@2", "R10@5")


def main() -> None:
    out = Path("resources/metrics")
    out.mkdir(parents=True, exist_ok=True)

    for task, row, alpha, result, log, minutes in RUNS:
        payload = {"task": task, "row": row, "rPU": 0.8, "seed": 0, "alpha": alpha,
                   **dict(zip(KEYS, result)), "source": f"log:{log}",
                   "commit": None, "wall_minutes": minutes}
        (out / f"{task}_{row}_rPU0.8_seed0.json").write_text(json.dumps(payload, indent=2))

    for task, result in CITED:
        payload = {"task": task, "row": "fs_pt_ft", "rPU": None, "seed": 0, "alpha": None,
                   **dict(zip(KEYS, result)), "source": "cite:han2021",
                   "commit": None, "wall_minutes": None}
        (out / f"{task}_fs_pt_ft_seed0.json").write_text(json.dumps(payload, indent=2))

    print(f"wrote {len(RUNS) + len(CITED)} files to {out}")


if __name__ == "__main__":
    main()
```

The two `pu_zero` rows and the `e_commerce` MAP/MRR values the logs do not record are set to `0.0`; only `R10@k` feeds Table 1. Before committing, confirm each `R10@k` triple against the published Table 1 in the paper PDF.

This refines the spec, which named a single `cited_fs_pt_ft.json`. One file per corpus is
used instead, so the cited row flows through the same `(task, row)` grouping as every other
row and needs no special case in the aggregator.

- [ ] **Step 2: Run the backfill**

Run: `python scripts/backfill_seed0.py`
Expected: `wrote 8 files to resources/metrics`

- [ ] **Step 3: Verify the aggregator reads them and warns about missing seeds**

Run: `python tools/aggregate_table.py 2>&1 | tail -20`
Expected: a table whose `R10@1` column reads `0.911`, `0.877`, `0.813`, `0.824` for Ubuntu — the published values — with warnings that each PU row has 1 of 3 seeds.

- [ ] **Step 4: Commit**

```bash
git add scripts/backfill_seed0.py resources/metrics
git commit -m "Backfill seed-0 metrics from the previous campaign with provenance"
```

---

### Task 9: Repository wiring and the equivalence check

The gate before any long job runs. If a check fails, **stop and report** — do not re-run the row to make the problem go away.

**Files:**
- Create: symlinks under `data/`, `FPT/`, `FineTuning/`
- Create: `scripts/check_equivalence.sh`

**Interfaces:**
- Consumes: the `--max-batches` and `--seed` flags from Task 5.
- Produces: confidence that the reused seed-0 metrics from Task 8 describe the new code path.

- [ ] **Step 1: Wire the data and checkpoints in**

```bash
ln -s /nfs/home/dasaro/research/innonext-BERT_FP/data/ubuntu_data/dataset_1M.pkl data/ubuntu_data/dataset_1M.pkl
ln -s /nfs/home/dasaro/research/innonext-BERT_FP/data/e_commerce_data/dataset_1M.pkl data/e_commerce_data/dataset_1M.pkl
ln -s /nfs/projects/innonext/models/BERT_FP_PU/PT_checkpoint FPT/PT_checkpoint
ln -s /nfs/projects/innonext/models/BERT_FP_PU/FT_checkpoint FineTuning/FT_checkpoint
```

- [ ] **Step 2: Verify the links resolve and are gitignored**

Run: `ls -lL data/ubuntu_data/dataset_1M.pkl FPT/PT_checkpoint/positive_only/ubuntu/checkpoint25/ | head -4 && git status --short`
Expected: the pickle and the checkpoint directory both listed; `git status` shows no new untracked paths.

- [ ] **Step 3: Produce the α̂ estimates**

The 30MB prior maps already exist on disk, so this reads them rather than re-extracting embeddings.

```bash
scripts/estimate_prior.sh ubuntu 0.8
scripts/estimate_prior.sh e_commerce 0.8
```

Expected: `resources/prior/alpha_hat_{ubuntu,e_commerce}_rPU0.8_{train,dev}.json` written, with `alpha_hat` near 0.435 for Ubuntu and 0.459 for E-Commerce — the values reported in Table 2.

- [ ] **Step 4: Write `scripts/check_equivalence.sh`**

```bash
#!/usr/bin/env bash
# Compares the refactored code's first-batch loss at seed 0 against the losses logged by
# the previous campaign. A match confirms model init, data order and alpha all line up,
# so the reused seed-0 metrics still describe this code path.
# Usage: scripts/check_equivalence.sh
set -uo pipefail

run() {
    local task="$1" objective="$2" prior="$3" expected="$4"
    local got
    got=$(python -u FineTuning/Response_selection.py \
              --task "$task" --rPU 0.8 --post-train positive_only \
              --objective "$objective" --prior "$prior" --seed 0 \
              --max-batches 1 --is_training 2>/dev/null \
          | grep -oP 'Batch\[0\] - loss: \K[0-9.]+' | head -1)

    if [[ "$got" == "$expected" ]]; then
        printf '  OK   %-12s %-14s %s\n' "$task" "${objective}/${prior}" "$got"
    else
        printf '  FAIL %-12s %-14s expected %s, got %s\n' "$task" "${objective}/${prior}" "$expected" "${got:-<none>}"
        return 1
    fi
}

status=0
run ubuntu     pu         exact     1.188617 || status=1
run ubuntu     pu         zero      1.269248 || status=1
run ubuntu     supervised exact     0.551310 || status=1
run e_commerce pu         exact     1.622780 || status=1
run e_commerce pu         zero      1.487587 || status=1
run e_commerce supervised exact     0.705367 || status=1

[[ $status -eq 0 ]] && echo "all six match" || echo "MISMATCH: stop and report before launching jobs"
exit $status
```

```bash
chmod +x scripts/check_equivalence.sh
```

- [ ] **Step 5: Run the equivalence check on a GPU node**

```bash
srun --ntasks 1 --cpus-per-task 16 --mem=64G --partition=A100 --gpus=3g.40gb:1 -t 2:00:00 \
     scripts/check_equivalence.sh
```

Expected: `all six match`.

**If any line reports FAIL, stop here and report to the user.** The first thing to suspect is an added or removed RNG consumer between `seed_everything` and the first forward pass, per the Global Constraints invariant. Do not proceed to Task 10.

- [ ] **Step 6: Commit**

```bash
git add scripts/check_equivalence.sh resources/prior
git commit -m "Add seed-0 equivalence check and persist the estimated priors"
```

---

### Task 10: Slurm job generation

**Files:**
- Modify: `scripts/finetune.sh`
- Create: `scripts/gen_jobs.py`
- Create: `jobs/` (18 `.job` files plus `submit_all.sh`, all generated)

**Interfaces:**
- Consumes: the CLI from Task 5.
- Produces: job files the user submits. This task does not run them.

- [ ] **Step 1: Rewrite `scripts/finetune.sh`**

```bash
#!/usr/bin/env bash
# PU or supervised fine-tuning (stage 2) and test-set evaluation.
# Usage: scripts/finetune.sh [ubuntu|e_commerce] [pu|supervised] [exact|zero|estimated] [seed] [rPU]
set -euo pipefail

TASK="${1:-ubuntu}"
OBJECTIVE="${2:-pu}"
PRIOR="${3:-exact}"
SEED="${4:-0}"
RPU="${5:-0.8}"

python -u FineTuning/Response_selection.py \
    --task "$TASK" \
    --rPU "$RPU" \
    --post-train positive_only \
    --objective "$OBJECTIVE" \
    --prior "$PRIOR" \
    --seed "$SEED" \
    --is_training
```

- [ ] **Step 2: Write `scripts/gen_jobs.py`**

```python
from __future__ import annotations

from pathlib import Path

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task}_{row}_seed{seed}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=msteams:dasaro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=A100
#SBATCH --gpus=3g.40gb:1
#SBATCH --output=logs/{row}/{task}_seed{seed}.out

source .venv/bin/activate
scripts/finetune.sh {task} {objective} {prior} {seed} 0.8
"""

# seed 0 is reused from the previous campaign for the three pre-existing rows
PLAN = [
    ("supervised_ft", "supervised", "exact", (1, 2)),
    ("pu_zero", "pu", "zero", (1, 2)),
    ("pu_exact", "pu", "exact", (1, 2)),
    ("pu_estimated", "pu", "estimated", (0, 1, 2)),
]
TASKS = ("ubuntu", "e_commerce")


def main() -> None:
    written = []
    for row, objective, prior, seeds in PLAN:
        for task in TASKS:
            for seed in seeds:
                path = Path("jobs") / row / f"{task}_seed{seed}.job"
                path.parent.mkdir(parents=True, exist_ok=True)
                Path(f"logs/{row}").mkdir(parents=True, exist_ok=True)
                path.write_text(TEMPLATE.format(task=task, row=row, objective=objective,
                                                prior=prior, seed=seed))
                written.append(path)

    submit = Path("jobs/submit_all.sh")
    submit.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n"
                      + "".join(f"sbatch {p}\n" for p in written))
    submit.chmod(0o755)
    print(f"wrote {len(written)} job files and {submit}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate and verify the matrix**

Run: `python scripts/gen_jobs.py && find jobs -name '*.job' | wc -l && cat jobs/pu_estimated/ubuntu_seed0.job`
Expected: `wrote 18 job files and jobs/submit_all.sh`, then `18`, then a job whose last line is `scripts/finetune.sh ubuntu pu estimated 0 0.8`.

- [ ] **Step 4: Confirm the seed split is right**

Run: `ls jobs/pu_exact jobs/pu_estimated`
Expected: `pu_exact` holds seeds 1 and 2 only (seed 0 is reused); `pu_estimated` holds seeds 0, 1 and 2.

- [ ] **Step 5: Commit**

```bash
git add scripts/finetune.sh scripts/gen_jobs.py jobs
git commit -m "Generate the 18-run three-seed job matrix"
```

---

### Task 11: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing downstream.

- [ ] **Step 1: Update the fine-tuning section**

Replace the `scripts/finetune.sh ... statistical_prior` examples with the four row invocations:

```bash
scripts/finetune.sh ubuntu supervised exact     0 0.8   # PU_PT + FS_FT
scripts/finetune.sh ubuntu pu         zero      0 0.8   # PU_PT+FT, alpha = 0
scripts/finetune.sh ubuntu pu         exact     0 0.8   # PU_PT+FT, alpha = alpha*
scripts/finetune.sh ubuntu pu         estimated 0 0.8   # PU_PT+FT, alpha = alpha-hat
```

Note that `--objective` picks the loss and `--post-train` picks the checkpoint, and that the old `--strategy` flag is gone.

- [ ] **Step 2: Add a three-seed reproduction section**

Document, in prose: run `scripts/estimate_prior.sh` for both corpora first, since `--prior estimated` reads its output; then `python scripts/gen_jobs.py && jobs/submit_all.sh`; then `python tools/aggregate_table.py` to write `results/table1.md`. State that seed 0 of the three pre-existing rows is reused from the earlier campaign and verified by `scripts/check_equivalence.sh`.

- [ ] **Step 3: State the three caveats for the paper**

Add a short subsection recording that variance is fine-tuning-only (post-training and the PU split are fixed), that α̂ is deterministic given the fixed checkpoint and so carries no seed variance, and that `cudnn.benchmark` stays enabled so runs are seed-controlled but not bitwise reproducible.

- [ ] **Step 4: Regenerate the reported-results tables**

Once all 18 runs land, replace the two tables under "Reported results" with the contents of `results/table1.md`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document the flag redesign and three-seed protocol"
```

---

## After the plan

Tasks 1–8, 10 and 11 need no GPU. Task 9 needs a GPU node and is the gate: nothing in the 234-GPU-hour campaign should be submitted until `scripts/check_equivalence.sh` prints `all six match`. The final `results/table1.md` and the paper's Table 1 are written only after all 18 jobs complete.
