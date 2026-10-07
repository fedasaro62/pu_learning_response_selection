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

parser.add_argument("--rPU",
                    default=None,
                    type=float,
                    choices=[0.8, 0.9],
                    help="The dataset used for training and test.")
parser.add_argument("--post_train",
                    default='positive_only',
                    type=str,
                    choices=['positive_only', 'positive_negative_unlabeled'],
                    help="The post-training method used for training the model.")

parser.add_argument("--strategy",
                    default='statistical_prior',
                    type=str,
                    choices=['statistical_prior', 'pos_neg_finetune'],
                    help="The fine-tuning objective: 'statistical_prior' is the PU loss of Eq. 7 with the "
                         "exact class prior alpha*, 'pos_neg_finetune' is the same loss with alpha=0.")

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
                    type=float,
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

dir_path = os.path.join(args.save_path, args.task)
if args.strategy == 'statistical_prior':
    from BERT_finetuning.statistical_prior import NeuralNetwork
elif args.strategy == 'pos_neg_finetune':
    from BERT_finetuning.pos_neg_finetune import NeuralNetwork
dir_path = os.path.join(dir_path, args.strategy)
suffix = ''
os.makedirs(dir_path, exist_ok=True)

if args.rPU is None:
    args.save_path = os.path.join(dir_path, 'bert.pt')
    args.score_file_path = os.path.join(dir_path, 'scorefile.txt')
else:
    args.save_path = os.path.join(dir_path, f'bert_rPU{args.rPU}{suffix}.pt')
    args.score_file_path = os.path.join(dir_path, f'scorefile_rPU{args.rPU}{suffix}.txt')

print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')

    if args.is_training==True:
        train_model(train,dev)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




