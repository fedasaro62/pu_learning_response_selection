from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FineTuning.BERT_finetuning.priors import PRIOR_DIR, estimated_prior_path, prior_map_path  # noqa: E402

STABLE_TAU_WINDOW = 5


@dataclass
class Args:
    task: str = "e_commerce"  # "ubuntu" | "douban" | "e_commerce"
    split: str = "train"  # "train" | "dev"
    rpu: float = 0.8
    save_path: str = "./FineTuning/FT_checkpoint/"
    score_transform: str = "cosine"  # "cosine" | "sigmoid"


FT_DATA = {
    "ubuntu": "data/ubuntu_data/dataset_1M.pkl",
    "douban": "data/douban_data/dataset_1M.pkl",
    "e_commerce": "data/e_commerce_data/dataset_1M.pkl",
}


def exact_alpha(y: np.ndarray, rpu: float) -> float:
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    n_labeled = round((1 - rpu) * n_pos)
    prior = (n_pos - n_labeled) / (n_pos + n_neg - n_labeled)
    return float(prior)


def build_scores(prior_map: dict, data: dict, rpu: float, score_transform: str) -> tuple[np.ndarray, np.ndarray]:
    cosine_sim = np.array(list(prior_map.values()), dtype=float)
    if score_transform == "sigmoid":
        z = (cosine_sim - np.mean(cosine_sim)) / (np.std(cosine_sim) + 1e-12)
        sigmoid_scores = 1 / (1 + np.exp(-z))
        prior_map_scores = {idx: float(score) for idx, score in zip(prior_map.keys(), sigmoid_scores)}
    else:
        prior_map_scores = {idx: float(score) for idx, score in zip(prior_map.keys(), cosine_sim)}

    pu_labels = np.asarray(data[f"y_pu_{rpu}"])
    all_indices = np.arange(len(pu_labels))
    pos_indices = all_indices[pu_labels == 1]
    unlab_indices = all_indices[pu_labels == -1]

    def get_prior_score(prior_map_local: dict, idx: int) -> float:
        return float(prior_map_local.get(str(int(idx)), prior_map_local.get(int(idx))))

    pos_scores = np.array([get_prior_score(prior_map_scores, i) for i in pos_indices], dtype=float)
    unlab_scores = np.array([get_prior_score(prior_map_scores, i) for i in unlab_indices], dtype=float)
    return pos_scores, unlab_scores


def estimate_pi_threshold(s_pos: np.ndarray, s_unk: np.ndarray, tau: float) -> float:
    p_unk = float(np.mean(s_unk > tau))
    p_match = float(np.mean(s_pos > tau))
    if p_match == 0.0:
        raise ValueError("No known positives exceed tau; lower tau.")
    return float(min(p_unk / p_match, 1.0))


def select_best_tau(
    s_pos: np.ndarray,
    s_unk: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    taus = np.arange(0.5, 0.91, 0.01)
    pi_hats = np.array([estimate_pi_threshold(s_pos, s_unk, tau=t) for t in taus], dtype=float)
    best_idx = int(np.argmin(np.abs(pi_hats - alpha)))
    best_tau = float(taus[best_idx])
    best_pi = float(pi_hats[best_idx])
    best_diff = float(best_pi - alpha)
    return best_tau, best_pi, best_diff, taus, pi_hats


def select_stable_tau(taus: np.ndarray, pi_hats: np.ndarray, window: int = 5) -> tuple[float, float]:
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    half = window // 2
    variances = np.full_like(pi_hats, fill_value=np.nan, dtype=float)
    for i in range(half, len(pi_hats) - half):
        segment = pi_hats[i - half : i + half + 1]
        variances[i] = float(np.var(segment))
    best_idx = int(np.nanargmin(variances))
    return float(taus[best_idx]), float(pi_hats[best_idx])


def write_estimate(prior_dir: Path, task: str, rpu: float, split: str, tau_star: float,
                   alpha_hat: float, alpha_exact: float, window: int, score_transform: str,
                   taus: np.ndarray, pi_hats: np.ndarray) -> Path:
    prior_dir.mkdir(parents=True, exist_ok=True)
    path = prior_dir / estimated_prior_path(task, rpu, split).name
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate alpha using stable tau.")
    parser.add_argument("--task", choices=["ubuntu", "douban", "e_commerce"], default=Args.task)
    parser.add_argument("--split", choices=["train", "dev"], default=Args.split)
    parser.add_argument("--rpu", type=float, default=Args.rpu)
    parser.add_argument("--save-path", default=Args.save_path)
    parser.add_argument("--score-transform", choices=["cosine", "sigmoid"], default=Args.score_transform)
    parsed = parser.parse_args()
    args = Args(
        task=parsed.task,
        split=parsed.split,
        rpu=parsed.rpu,
        save_path=parsed.save_path,
        score_transform=parsed.score_transform,
    )

    print(args.task)

    with open(FT_DATA[args.task], "rb") as f:
        train, dev, _ = pickle.load(f, encoding="ISO-8859-1")

    prior_map_base = f"{args.save_path}/positive_only/{args.task}/fixed_prior/"
    data = {"train": train, "dev": dev}[args.split]
    with open(prior_map_path(prior_map_base, args.rpu, args.split), "r") as f:
        prior_map = json.load(f)

    taus = np.arange(0.5, 0.91, 0.01)
    alpha = exact_alpha(np.asarray(data["y"]), args.rpu)
    pos_scores, unlab_scores = build_scores(prior_map, data, args.rpu, args.score_transform)
    pi_hats = np.array(
        [estimate_pi_threshold(pos_scores, unlab_scores, tau=t) for t in taus],
        dtype=float,
    )
    stable_tau, stable_pi = select_stable_tau(taus, pi_hats, window=STABLE_TAU_WINDOW)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(5.5, 3))
    sns.lineplot(x=taus, y=pi_hats, linewidth=2, color="tab:blue")
    plt.axhline(alpha, linestyle="--", color="tab:red", linewidth=2)
    plt.text(
        x=float(taus[15]),
        y=float(alpha),
        s=rf"$\alpha^* = {alpha:.2f}$",
        color="tab:red",
        fontsize=17,
        va="bottom",
        ha="right",
    )
    plt.axvline(stable_tau, linestyle="--", color="black", linewidth=2)
    plt.text(
        x=float(stable_tau) + 0.06,
        y=float(pi_hats.max()) - 0.7,
        s=rf"$\mathbf{{{stable_tau:.2f}}}$",
        color="black",
        fontsize=16,
        va="bottom",
        ha="center",
    )
    plt.xlabel("τ", fontsize=17)
    plt.ylabel(r"$\hat{\alpha}$", fontsize=17)
    plt.title("Train" if args.split == "train" else "Dev", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.0)
    plt.tight_layout(pad=0.6)
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/pi_hat_vs_tau_{args.split}_{args.task}_rPU{args.rpu}.png", dpi=300)

    written = write_estimate(PRIOR_DIR, args.task, args.rpu, args.split,
                             stable_tau, stable_pi, alpha, STABLE_TAU_WINDOW, args.score_transform,
                             taus, pi_hats)
    print(f"wrote {written}")

    print(f"[{args.split.upper()}] alpha={alpha:.4f}")
    print(
        f"[{args.split.upper()}] stable tau={stable_tau:.2f}, pi_hat={stable_pi:.4f}, diff={stable_pi - alpha:+.4f}"
    )


if __name__ == "__main__":
    main()
