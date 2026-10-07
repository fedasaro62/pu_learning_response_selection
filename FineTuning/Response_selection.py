import time
import argparse
import pickle
import os
from setproctitle import setproctitle

setproctitle('BERT_FP')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
print("CUDA available:", torch.cuda.is_available())

#Dataset path.
FT_data={
    'ubuntu': 'data/ubuntu_data/dataset_1M.pkl',
    'douban': 'data/douban_data/dataset_1M.pkl',
    'e_commerce': 'data/e_commerce_data/dataset_1M.pkl'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")

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

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="The batch size.")
parser.add_argument("--num_workers",
                    default=16,
                    type=int,
                    help="The number of workers.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs",
                    default=2,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./FineTuning/FT_checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="./FineTuning/scorefile.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
args = parser.parse_args()

from BERT_finetuning.model import CHECKPOINTS, NeuralNetwork, seed_everything
from BERT_finetuning.priors import resolve_prior
from run_config import row_tag, run_paths, validate_args, write_metrics

validate_args(args.objective, args.post_train, args.rPU)
args.post_train_ckpt = args.post_train_ckpt or CHECKPOINTS[args.task]
row = row_tag(args.objective, args.prior, args.post_train)
args.save_path, args.score_file_path, args.metrics_path = run_paths(args.task, row, args.seed, args.rPU)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    result = model.evaluate(test, is_test=True)
    write_metrics(args.metrics_path, task=args.task, row=row, rPU=args.rPU, seed=args.seed,
                  alpha=args.prior_value, result=result, source="run",
                  commit=os.popen("git rev-parse --short HEAD").read().strip() or None,
                  wall_minutes=(time.time() - start) / 60)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    args.prior_value = resolve_prior(args.objective, args.prior, args.task, args.rPU, train['y'])
    print(f"row={row} objective={args.objective} prior={args.prior} alpha={args.prior_value}")
    seed_everything(args.seed)

    if args.is_training:
        train_model(train, dev)
        if args.max_batches is None:
            test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




