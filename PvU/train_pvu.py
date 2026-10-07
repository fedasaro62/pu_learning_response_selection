import time
import argparse
import pickle
import os
from setproctitle import setproctitle
from BERT_pvu_finetuning import NeuralNetwork

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

parser.add_argument("--rPU",
                    default=None,
                    type=float,
                    choices=[0.8, 0.9, None],
                    help="The dataset used for training and test.")
parser.add_argument("--num_estimate_per_epoch",
                    default=10,
                    type=int,
                    help="Number of alpha estimates per epoch.")

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")
parser.add_argument("--subsample",
                    default=None,
                    type=float,
                    help="Subsample fraction for dev set.")

parser.add_argument("--batch_size",
                    default=32,
                    type=float,
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
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./PvU/pvu_checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")
args = parser.parse_args()

args.save_path = os.path.join(args.save_path, f"{args.task}_rPU{args.rPU}_new")
os.makedirs(args.save_path, exist_ok=True)
# args.save_path = os.path.join(dir_path, f'pvu_finetune_rPU{args.rPU}.log')

print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(os.path.join(args.save_path, "bert.pt"))
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if args.is_training==True:
        train_model(train,dev)
        # test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




