from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FineTuning")))

from BERT_finetuning.data import BERTDataset  # noqa: E402

FT_DATA = {
    "ubuntu": "data/ubuntu_data/dataset_1M.pkl",
    "e_commerce": "data/e_commerce_data/dataset_1M.pkl",
}
FT_MODEL = {
    "ubuntu": "bert-base-uncased",
    "e_commerce": "bert-base-chinese",
}
CHECKPOINTS = {
    "ubuntu": 25,
    "e_commerce": 34,
}


class DatasetArgs:
    """minimal stand-in for the argparse namespace BERTDataset expects."""

    def __init__(self, task: str, rpu: float | None) -> None:
        self.task = task
        self.rPU = rpu
        self.do_lower_case = True


def build_encoder(task: str, checkpoint: str | None, device: torch.device) -> tuple[BertModel, BertTokenizer]:
    model_name = FT_MODEL[task]
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    tokenizer.add_special_tokens({"eos_token": "[eos]"})

    model = BertModel(BertConfig.from_pretrained(model_name))
    model.resize_token_embeddings(len(tokenizer))
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval(), tokenizer


def pair_cosine_similarities(
    model: BertModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """mean-pool the last hidden layer over context and response spans, then cosine-compare them."""
    context_embs: list[np.ndarray] = []
    response_embs: list[np.ndarray] = []
    labels: list[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids, input_mask, segment_ids, batch_y, _, lenidx, _ = batch
            hidden = model(
                input_ids.to(device),
                attention_mask=input_mask.to(device),
                token_type_ids=segment_ids.to(device),
            ).last_hidden_state

            for i in range(len(input_ids)):
                ctx_end = lenidx[i][0].item()
                rsp_end = lenidx[i][1].item()
                context_embs.append(hidden[i, 1:ctx_end, :].mean(0).cpu().numpy())
                response_embs.append(hidden[i, ctx_end + 1 : rsp_end, :].mean(0).cpu().numpy())

            labels.extend(batch_y.cpu().numpy())

    ctx = np.asarray(context_embs)
    rsp = np.asarray(response_embs)
    ctx /= np.linalg.norm(ctx, axis=1, keepdims=True)
    rsp /= np.linalg.norm(rsp, axis=1, keepdims=True)
    return np.sum(ctx * rsp, axis=1), np.asarray(labels)


def plot(similarities: np.ndarray, labels: np.ndarray, title: str, out_path: str) -> None:
    match = similarities[labels == 1]
    unmatch = similarities[labels == 0]
    mean_diff = match.mean() - unmatch.mean()

    plt.figure(figsize=(10, 6))
    plt.hist(match, bins=50, alpha=0.6, density=True, color="green", label=f"Match, μ={match.mean():.3f}")
    plt.hist(unmatch, bins=50, alpha=0.6, density=True, color="red", label=f"Unmatch, μ={unmatch.mean():.3f}")
    plt.axvline(match.mean(), color="green", linestyle="--", linewidth=2, label=f"Δμ = {mean_diff:.3f}")
    plt.axvline(unmatch.mean(), color="red", linestyle="--", linewidth=2)
    plt.xlabel("Cosine Similarity", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)

    print(f"Label distribution: {Counter(labels.tolist())}")
    print(f"Match   mean={match.mean():.4f} std={match.std():.4f} n={len(match)}")
    print(f"Unmatch mean={unmatch.mean():.4f} std={unmatch.std():.4f} n={len(unmatch)}")
    print(f"Delta mu={mean_diff:.4f}")
    print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot matched vs unmatched cosine similarity distributions.")
    parser.add_argument("--task", choices=list(FT_DATA), default="ubuntu")
    parser.add_argument("--rPU", type=float, default=0.8, help="PU ratio of the post-training checkpoint")
    parser.add_argument("--zero-shot", action="store_true", help="use pre-trained BERT instead of a checkpoint")
    parser.add_argument("--checkpoint", default=None, help="explicit path to a post-trained encoder")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="results/density/cosine_similarity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = None
    if not args.zero_shot:
        checkpoint = args.checkpoint or (
            f"FPT/PT_checkpoint/positive_only/{args.task}/checkpoint{CHECKPOINTS[args.task]}/bert_rPU{args.rPU}.pt"
        )

    model, tokenizer = build_encoder(args.task, checkpoint, device)

    with open(FT_DATA[args.task], "rb") as f:
        train, _, _ = pickle.load(f, encoding="ISO-8859-1")

    dataset = BERTDataset(DatasetArgs(args.task, args.rPU), train, tokenizer)
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    dataloader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)

    similarities, labels = pair_cosine_similarities(model, dataloader, device)

    stage = "zero_shot" if args.zero_shot else f"post_trained_rPU{args.rPU}"
    plot(
        similarities,
        labels,
        title=f"{args.task} ({'zero-shot' if args.zero_shot else 'post-trained'})",
        out_path=os.path.join(args.out_dir, f"{args.task}_similarity_distribution_{stage}.png"),
    )


if __name__ == "__main__":
    main()
