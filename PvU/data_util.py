import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import  RandomSampler
from torch.optim import AdamW
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
# from Metrics import Metrics

FT_model={
    'ubuntu': 'bert-base-uncased',
    'douban': 'bert-base-chinese',
    'e_commerce': 'bert-base-chinese'
}

ORIG_NAMES = {
    'ubuntu': 'ubuntu25',
    'douban': 'douban27',
    'e_commerce': 'e_commerce34'
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, args, data, tokenizer, split='train'):
        self.data = data
        self.args = args
        self.bert_tokenizer = tokenizer
        self.split = split
    
    def __len__(self):
        return len(self.data['cr'])

    def convert_examples_to_features(self, item):
        """ Loads a data file into a list of `InputBatch`s """
        ex_index = item
        input_ids = self.data['cr'][item]

        sep = input_ids.index(self.bert_tokenizer.sep_token_id)
        context = input_ids[:sep]
        response = input_ids[sep + 1:]
        _truncate_seq_pair(context, response, 253)

        context_len = len(context)

        input_ids = [self.bert_tokenizer.cls_token_id] + context + [self.bert_tokenizer.sep_token_id] + response + [
            self.bert_tokenizer.sep_token_id]
        segment_ids = [0] * (context_len + 2)  # context
        segment_ids += [1] * (len(input_ids) - context_len - 2)  # #response
        
        lenidx = [1 + context_len, len(input_ids) - 1]

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = 256 - len(input_ids)

        if (padding_length > 0):
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)

        return input_ids, input_mask, segment_ids, lenidx, ex_index


class BERTPosData(BERTDataset):
    """Positive (labeled) BERT dataset for PU learning"""
    def __init__(self, args, data, tokenizer, indices, split='train'):
        super().__init__(args, data, tokenizer, split)
        self.indices = indices  # Indices of positive samples in original data
        self.pu_label = 0
        self.true_label = 0  # All labeled are positive
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item):
        actual_idx = self.indices[item]
        
        input_ids, input_mask, segment_ids, lenidx, _ = self.convert_examples_to_features(actual_idx)
        
        cur_tensors = (
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(segment_ids),
            torch.tensor(self.true_label, dtype=torch.float),  # True label
            torch.tensor(self.pu_label, dtype=torch.float),  # PU label = 0 (positive)
            torch.tensor(lenidx),
            torch.tensor(item, dtype=torch.long)  # Index within positive dataset
        )
        
        return cur_tensors


class BERTUnlabelData(BERTDataset):
    """Unlabeled BERT dataset for PU learning (contains both positive and negative)"""
    def __init__(self, args, data, tokenizer, indices, split='train'):
        super().__init__(args, data, tokenizer, split)
        self.indices = indices  # Indices of unlabeled samples in original data
        self.pu_label = 1  # All samples are unlabeled
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, item):
        actual_idx = self.indices[item]
        
        input_ids, input_mask, segment_ids, lenidx, _ = self.convert_examples_to_features(actual_idx)
        
        true_label = int(self.data['y'][actual_idx] == 0)
        
        cur_tensors = (
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(segment_ids),
            torch.tensor(true_label, dtype=torch.float),  # True label
            torch.tensor(self.pu_label, dtype=torch.float),  # PU label = 1 (unlabeled)
            torch.tensor(lenidx),
            torch.tensor(item, dtype=torch.long)  # Index within unlabeled dataset
        )
        
        return cur_tensors


def get_BERT_PU_datasets(args, data, tokenizer, split='train'):
    assert args.rPU is not None, "rPU must be provided"
    
    pu_key = f'y_pu_{args.rPU}'
    assert pu_key in data, f"PU labels not found: {pu_key}"
    
    pos_indices = []
    unlabel_indices = []
    
    for idx in range(len(data['cr'])):
        if data[pu_key][idx] == 1:  # Labeld (positive)
            pos_indices.append(idx)
        else:  # Unlabeled -1
            unlabel_indices.append(idx)
    
    print(f"{split.capitalize()} split - Positive: {len(pos_indices)}, Unlabeled: {len(unlabel_indices)}")
    
    pos_dataset = BERTPosData(args, data, tokenizer, pos_indices, split=split)
    unlabel_dataset = BERTUnlabelData(args, data, tokenizer, unlabel_indices, split=split)

    if split == 'dev' and args.subsample is not None:
        pos_dataset = Subset(pos_dataset, np.random.choice(len(pos_dataset), int(len(pos_dataset)*args.subsample), replace=False))
        unlabel_dataset = Subset(unlabel_dataset, np.random.choice(len(unlabel_dataset), int(len(unlabel_dataset)*args.subsample), replace=False))
        print(f"Subsampled {split} split - Positive: {len(pos_dataset)}, Unlabeled: {len(unlabel_dataset)}")
    
    return pos_dataset, unlabel_dataset

def get_BERT_PU_dataloaders(args, data, tokenizer, split='train'):
    beta = round((1-args.rPU)/2, 2)
    print(f"{split.capitalize()} Beta: {beta}")

    pos_dataset, unlabel_dataset = get_BERT_PU_datasets(args, data, tokenizer, split=split)

    shuffle = split == 'train'
    p_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=int(args.batch_size), \
            shuffle=shuffle, num_workers=args.num_workers)
    u_loader = torch.utils.data.DataLoader(unlabel_dataset, batch_size=int(args.batch_size*(1-beta)/beta), \
        shuffle=shuffle, num_workers=args.num_workers)
    return p_loader, u_loader


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()