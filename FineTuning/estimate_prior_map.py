import argparse
import os
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Subset, RandomSampler, DataLoader
from BERT_finetuning.data import BERTDataset
from BERT_finetuning.model import NeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    'ubuntu': 25,
    'douban': 27,
    'e_commerce': 34
}
FT_data={
    'ubuntu': 'data/ubuntu_data/dataset_1M.pkl',
    'douban': 'data/douban_data/dataset_1M.pkl',
    'e_commerce': 'data/e_commerce_data/dataset_1M.pkl'
}
FT_model={
    'ubuntu': 'bert-base-uncased',
    'douban': 'bert-base-chinese',
    'e_commerce': 'bert-base-chinese'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate prior map for fine-tuned BERT")
    parser.add_argument("--task", choices=list(CHECKPOINTS.keys()), default="ubuntu", help="Dataset/task to use")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train", help="Data split to use for estimation")
    parser.add_argument("--rPU", type=float, default=0.9, help="PU ratio used for checkpoint naming")
    parser.add_argument("--post-train", default="positive_only", choices=["positive_only", "positive_and_unlabeled", "zero_shot"], help="Post-train setting name")
    parser.add_argument("--post-train-ckpt", type=int, default=None, help="Checkpoint index for post-train stage")
    parser.add_argument("--save-path", default="./FineTuning/FT_checkpoint/", help="Base directory to save prior map")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of DataLoader workers")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to use for estimation")
    parser.add_argument("--do_lower_case", action='store_true', default=True, help="Set this flag if you are using an uncased model.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.post_train_ckpt is None:
        args.post_train_ckpt = CHECKPOINTS[args.task]

    args.score_file_path = ""
    args.objective = "pu"
    args.prior_value = None
    args.save_path = os.path.join(args.save_path, args.post_train, args.task, 'fixed_prior', f'bert_rPU{args.rPU}.pt')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    model = NeuralNetwork(args=args)

    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')
    data = {'train': train, 'dev': dev, 'test': test}[args.split]

    n_samples = args.num_samples if args.num_samples is not None else len(data['y'])
    random.seed(42)
    full_dataset = BERTDataset(args, data, model.bert_tokenizer)
    subset_indices = random.sample(range(len(full_dataset)), n_samples)
    dataset = Subset(full_dataset, subset_indices)
    print(f"Using {len(dataset)} samples for prior estimation.")
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    
    model.extract_similarities(dataloader, split=args.split)

if __name__ == "__main__":
    main()