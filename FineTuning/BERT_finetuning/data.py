from __future__ import annotations

import torch
from torch.utils.data import Dataset

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label, pu_label, lenidx, sample_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.pu_label = pu_label # new
        self.lenidx = lenidx
        self.sample_idx = sample_idx # new


class BERTDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data = data
        self.args = args
        self.bert_tokenizer = tokenizer

        self.PU = False
        if args.rPU is not None:
            self.PU = True
            if f'y_pu_{args.rPU}' in self.data:
                self.split = 'train'
            else:
                print("PU labels not found in the dataset. You are using dev or test split.")
                self.split = 'dev/test'
    
    def __len__(self):
        return len(self.data['cr'])

    def __getitem__(self, item):
        cur_features = self.convert_examples_to_features(item)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.label, dtype=torch.float),
                       torch.tensor(cur_features.pu_label, dtype=torch.float),
                       torch.tensor(cur_features.lenidx),
                       torch.tensor(cur_features.sample_idx, dtype=torch.long)
                       )

        return cur_tensors

    def convert_examples_to_features(self, item):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        ex_index = item
        input_ids = self.data['cr'][item]


        sep = input_ids.index(self.bert_tokenizer.sep_token_id)
        context = input_ids[:sep]
        response = input_ids[sep + 1:]
        _truncate_seq_pair(context, response, 253)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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

        if self.PU and self.split == 'train':
            pu_label = self.data[f'y_pu_{self.args.rPU}'][item]
        else:
            pu_label = 255

        features = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label=self.data['y'][item],
            pu_label=pu_label,
            lenidx=lenidx,
            sample_idx=ex_index,
        )
        return features


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
