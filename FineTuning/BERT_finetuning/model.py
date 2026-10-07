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
from .priors import prior_map_path

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
    special = {
        'zero_shot': lambda: None,
        'original': lambda: f"./FPT/PT_checkpoint/original/{task}{ckpt}/bert.pt",
    }
    if post_train in special:
        return special[post_train]()
    return f"./FPT/PT_checkpoint/{post_train}/{task}/checkpoint{ckpt}/bert_rPU{rPU}.pt"



class NeuralNetwork(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        self.bert_config = config_class.from_pretrained(FT_MODEL[args.task],num_labels=1)
        self.bert_tokenizer = BertTokenizer.from_pretrained(FT_MODEL[args.task],do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert_model = model_class.from_pretrained(FT_MODEL[args.task],config=self.bert_config)

        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        
        checkpoint = post_trained_checkpoint(args.post_train, args.task, args.post_train_ckpt, args.rPU)
        if checkpoint is None:
            print("zero-shot setting: using pre-trained BERT without loading a post-trained checkpoint")
        else:
            self.bert_model.bert.load_state_dict(
                state_dict=torch.load(checkpoint, map_location=self.device)
            )

        self.bert_model = self.bert_model.to(self.device)

        self.prior: float | None = args.prior_value

    def forward(self):
        raise NotImplementedError

    def extract_similarities(self, dataloader, split=None)->dict:
        """
        Returns mapping of sample indices to sigmoid-transformed similarity scores.
        """
        dirpath = os.path.dirname(self.args.save_path)
        prior_file = str(prior_map_path(dirpath, self.args.rPU, split))
        if os.path.exists(prior_file):
            print(f"Loading prior map from {prior_file}...")
            with open(prior_file, 'r') as f:
                prior_map = json.load(f)
            return prior_map
        
        context_embs, response_embs, sample_indices = [], [], []
        
        print("Extracting embeddings to estimate prior...")
        self.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, total=len(dataloader), desc="Estimating prior"):
                batch_ids, batch_mask, batch_seg, batch_y, batch_pu_y, batch_len, batch_idxs = (item.to(self.device) for item in data)
                
                outputs = self.bert_model(batch_ids, batch_mask, batch_seg, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                
                for i in range(len(batch_ids)):
                    ctx_end = batch_len[i][0].item()
                    rsp_end = batch_len[i][1].item()
                    
                    ctx_emb = last_hidden_states[i, 1:ctx_end, :].mean(0).cpu().numpy()
                    rsp_emb = last_hidden_states[i, ctx_end+1:rsp_end, :].mean(0).cpu().numpy()
                    
                    context_embs.append(ctx_emb)
                    response_embs.append(rsp_emb)

                sample_indices.extend(batch_idxs.cpu().numpy().tolist())
        
        context_embs = np.array(context_embs)
        response_embs = np.array(response_embs)
        
        context_norm = context_embs / np.linalg.norm(context_embs, axis=1, keepdims=True)
        response_norm = response_embs / np.linalg.norm(response_embs, axis=1, keepdims=True)
        cosine_sim = np.sum(context_norm * response_norm, axis=1)
        
        # cosine_sim_standardized = (cosine_sim - np.mean(cosine_sim)) / np.std(cosine_sim)
        # sigmoid_scores = 1 / (1 + np.exp(-cosine_sim_standardized))
        prior_map = {idx: float(score) for idx, score in zip(sample_indices, cosine_sim)}

        # print(f"\nPrior - Sigmoid scores statistics:")
        # print(f"  Mean: {np.mean(sigmoid_scores):.4f}")
        # print(f"  Min: {np.min(sigmoid_scores):.4f}")
        # print(f"  Max: {np.max(sigmoid_scores):.4f}")

        with open(prior_file, 'w') as f:
            json.dump(prior_map, f)
        
        return prior_map

    def train_step(self, i, data):
        batch_ids, batch_mask, batch_seg, batch_y, batch_pu_y, batch_len, batch_idxs = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()

        output = self.bert_model(batch_ids, batch_mask, batch_seg)
        logits = torch.sigmoid(output[0]).squeeze()

        loss = (supervised_loss(logits, batch_y) if self.prior is None
                else pu_loss(logits, batch_pu_y, self.prior))

        loss.backward()
        self.optimizer.step()

        if i % 100 == 0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
                                                                batch_y.size(0)))
        return loss

    def fit(self, train, dev):  

        if torch.cuda.is_available(): self.cuda()

        dataset = BERTDataset(self.args, train, self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=self.args.num_workers)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0

            self.train()
            for i, data in tqdm(enumerate(dataloader)): 
                if epoch >= 2 and self.patience >= 3:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.args.save_path))
                    self.adjust_learning_rate()
                    self.patience = 0

                loss = self.train_step(i, data)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()

                if self.args.max_batches is not None and i + 1 >= self.args.max_batches:
                    print(f"stopping after {i + 1} batches (--max-batches)")
                    return
            cnt = len(train['y']) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss / cnt))

            self.evaluate(dev)

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)

    def evaluate(self, dev, is_test=False):
        y_pred = self.predict(dev)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, dev['y']):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )
        if is_test == False and self.args.task !='ubuntu':
            self.metrics.segment = 2
        else:
            self.metrics.segment = 10
        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:", result[3], "\t",
              "R2:", result[4], "\t",
              "R5:", result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + \
                self.best_result[5]:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:", self.best_result[3], "\t",
                  "R2:", self.best_result[4], "\t",
                  "R5:", self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1
        return result

    def predict(self, dev):
        self.eval()
        y_pred = []
        dataset = BERTDataset(self.args, dev, self.bert_tokenizer)
        dataloader = DataLoader(dataset, batch_size=400)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_ids, batch_mask, batch_seg, batch_y, batch_pu_y, batch_len, _ = (item.cuda() for item in data)
                output = self.bert_model(batch_ids, batch_mask, batch_seg)
                logits = torch.sigmoid(output[0]).squeeze()
                if logits.dim() == 0:  # Single sample (scalar)
                    logits = logits.unsqueeze(0)

            if i % 100 == 0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0))) 
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred

    def load_model(self, path):
        self.load_state_dict(
            state_dict=torch.load(path, map_location=self.device)
        )
        if torch.cuda.is_available(): self.cuda()
