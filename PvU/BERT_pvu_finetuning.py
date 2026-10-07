import os
import csv
import torch
import torch.nn as nn
import torch.nn.utils as utils
import logging
import numpy as np

from torch.optim import AdamW
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
# from Metrics import Metrics
from data_util import get_BERT_PU_dataloaders
from utils import progress_bar
from estimators import BBE_estimator, dedpul, dedpul_acc, estimator_CM_EN, scott_estimator

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
logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_valid_estimate = 0.0
        # self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        print("Set num_labels=2")
        self.bert_config = config_class.from_pretrained(FT_model[args.task],num_labels=2)
        self.bert_tokenizer = BertTokenizer.from_pretrained(FT_model[args.task],do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert_model = model_class.from_pretrained(FT_model[args.task],config=self.bert_config)

        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        
        checkpoit_idx = {"ubuntu": 25, "douban": 27, "e_commerce": 34}
        if args.rPU is None:
            self.bert_model.bert.load_state_dict(state_dict=torch.load(f"./FPT/PT_checkpoint/original/{ORIG_NAMES[args.task]}/bert.pt", map_location=self.device))
        else:
            self.bert_model.bert.load_state_dict(state_dict=torch.load(f"./FPT/PT_checkpoint/positive_only/{args.task}/checkpoint{checkpoit_idx[args.task]}/bert_rPU{args.rPU}.pt", map_location=self.device))
        
        self.bert_model = self.bert_model.to(self.device)

    def forward(self):
        raise NotImplementedError
    
    def estimate_alpha(self, valid_acc, pos_probs, unlabeled_probs, unlabeled_targets, epoch=None, i=None, outfile:str=None):
        scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
        bbe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
        EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
        dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)
        dedpul_probs_unlabeled = dedpul_probs[len(pos_probs):]
        dedpul_accuracy = dedpul_acc(dedpul_probs_unlabeled, unlabeled_targets) * 100.0

        if outfile is not None:
            with open(outfile, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([epoch, i, valid_acc, dedpul_accuracy,\
                    dedpul_estimate, EN_estimate, scott_mpe_estimator, bbe_estimate])
        return bbe_estimate

    def exact_alpha(self, y)->float:
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        n_labeled = round((1 - self.args.rPU) * n_pos)
        prior = (n_pos - n_labeled) / (n_pos + n_neg - n_labeled)
        # print(f"Estimated prior: {prior:.4f}")
        return prior

    def train_step(self, i, p_data, u_data):
        p_batch_ids, p_batch_mask, p_batch_seg, p_batch_y, p_batch_pu_y, p_batch_len, p_batch_idxs = \
            (item.to(self.device) for item in p_data)
        u_batch_ids, u_batch_mask, u_batch_seg, u_batch_y, u_batch_pu_y, u_batch_len, u_batch_idxs = \
            (item.to(self.device) for item in u_data)

        self.optimizer.zero_grad()

        batch_ids = torch.cat([p_batch_ids, u_batch_ids], dim=0)
        batch_mask = torch.cat([p_batch_mask, u_batch_mask], dim=0)
        batch_seg = torch.cat([p_batch_seg, u_batch_seg], dim=0)

        output = self.bert_model(batch_ids, batch_mask, batch_seg)

        p_output = output[0][:len(p_batch_ids)]
        u_output = output[0][len(p_batch_ids):]

        p_loss = self.loss_func(p_output, p_batch_pu_y.long()).mean()
        u_loss = self.loss_func(u_output, u_batch_pu_y.long()).mean()

        loss = (p_loss + u_loss) / 2

        loss.backward()
        self.optimizer.step()

        if i % 100 == 0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
                                                                len(p_batch_ids) + len(u_batch_ids)))
        return loss
    
    def fit(self, train, dev):  

        # if torch.cuda.is_available(): self.cuda()
        if torch.cuda.is_available(): 
            self.cuda()
            torch.cuda.empty_cache()
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

        p_trainloader, u_trainloader = get_BERT_PU_dataloaders(self.args, train, self.bert_tokenizer, split='train')
        p_validloader, u_validloader = get_BERT_PU_dataloaders(self.args, dev, self.bert_tokenizer, split='dev')

        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        beta = round((1-self.args.rPU)/2, 2)
        train_alpha = self.exact_alpha(train['y'])
        valid_alpha = self.exact_alpha(dev['y'])

        csv_path = os.path.join(self.args.save_path, 'true_alpha.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Split', 'True Alpha', 'Beta'])
            csv_writer.writerow(['Train', train_alpha, beta])
            csv_writer.writerow(['Valid', valid_alpha, beta])

        csv_path = os.path.join(self.args.save_path, 'dev_estimate_alpha.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Epoch", "Iteration", "Val_Acc", "DedPUL_Acc", "DedPUL_Alpha", "EN_Alpha", "Scott_Alpha", "BBE_Alpha"])

        num_steps_per_epoch = len(p_trainloader.dataset) // self.args.batch_size + 1
        estimate_steps = [
            int(i * num_steps_per_epoch / self.args.num_estimate_per_epoch)
            for i in range(1, self.args.num_estimate_per_epoch + 1)  # Start from 1, not 0
        ]
        estimate_every = num_steps_per_epoch // self.args.num_estimate_per_epoch
        print("\nNumber of steps per epoch: ", num_steps_per_epoch)
        print('Estimating every {} steps'.format(estimate_every))
        print('Estimate steps:', estimate_steps)
        
        valid_estimate = 0.0
        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            
            avg_loss = 0
            self.train()
            for i, (p_data, u_data) in enumerate(tqdm(zip(p_trainloader, u_trainloader))):
                if epoch >= 2 and self.patience >= 3:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(os.path.join(self.args.save_path, "bert.pt")))
                    self.adjust_learning_rate()
                    self.patience = 0

                loss = self.train_step(i, p_data, u_data)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
                
                if i in estimate_steps:
                    print(f"\n--- Estimation at epoch {epoch}, step {i} ---")
                    threshold = 0.5*beta/(beta + (1-beta)*valid_estimate)
                    print(f"Threshold: {threshold:.4f}")
                    
                    valid_acc, unlabeled_probs, unlabeled_targets = self.validate(
                        u_validloader, threshold, epoch=epoch, step=i, return_probs=True
                    )
                    pos_probs = self.p_probs(p_validloader)
                    valid_estimate = self.estimate_alpha(valid_acc, pos_probs, unlabeled_probs, unlabeled_targets, 
                                        epoch=epoch, i=i, outfile=csv_path)
                    
                    print(f"Validation Accuracy: {valid_acc:.2f}%")
                    print(f"Alpha estimate: {valid_estimate:.4f}")

                    best_error = abs(self.best_valid_estimate - valid_alpha)
                    error = abs(valid_estimate - valid_alpha)
                    if error < best_error:
                        self.best_valid_estimate = valid_estimate
                        print("New best alpha estimate error: {:.4f}".format(error))
                        torch.save(self.state_dict(), os.path.join(self.args.save_path, "bert_best.pt"))

                avg_loss += loss.item()
                
            print("Average loss:{:.6f} ".format(avg_loss / num_steps_per_epoch))
        
        print("\n--- Save Last Model ---")
        torch.save(self.state_dict(), os.path.join(self.args.save_path, "bert_last.pt"))

        print("\n--- Final Alpha estimation on Train Data ---")
        csv_path = os.path.join(self.args.save_path, 'train_estimate_alpha.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Epoch", "Iteration", "Val_Acc", "DedPUL_Acc", "DedPUL_Alpha", "EN_Alpha", "Scott_Alpha", "BBE_Alpha"])
        
        self.load_state_dict(torch.load(os.path.join(self.args.save_path, "bert_best.pt")))
        pos_probs = self.p_probs(p_trainloader)
        unlabeled_probs, unlabeled_targets = self.u_probs(u_trainloader)
        train_estimate_best = self.estimate_alpha('N/A', pos_probs, unlabeled_probs, unlabeled_targets, epoch='Best', i='Best', outfile=csv_path)

        self.load_state_dict(torch.load(os.path.join(self.args.save_path, "bert_last.pt")))
        pos_probs = self.p_probs(p_trainloader)
        unlabeled_probs, unlabeled_targets = self.u_probs(u_trainloader)
        train_estimate_last = self.estimate_alpha('N/A', pos_probs, unlabeled_probs, unlabeled_targets, epoch='Last', i='Last', outfile=csv_path)
        
        print(f"Train Alpha estimates (best model): {train_estimate_best:.4f}")
        print(f"Train Alpha estimates (last model): {train_estimate_last:.4f}")

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)

    
    def validate(self, u_validloader, threshold, epoch=None, step=None, show_bar=True, separate=False, return_probs=False):
        if epoch is not None and step is not None and show_bar:
            print('Validation at Epoch {} Step {}'.format(epoch + 1, step))
        else:
            print('Validation...')
        
        self.eval()

        test_loss = 0
        correct = 0
        total = 0

        pos_correct = 0
        neg_correct = 0

        pos_total = 0
        neg_total = 0

        all_probs = [] if return_probs else None
        all_targets = [] if return_probs else None

        with torch.no_grad():
            for batch_idx, u_data in enumerate(u_validloader):
                u_batch_ids, u_batch_mask, u_batch_seg, u_batch_y, u_batch_pu_y, u_batch_len, u_batch_idxs = \
                    (item.to(self.device) for item in u_data)
                u_output = self.bert_model(u_batch_ids, u_batch_mask, u_batch_seg)
                
                logits = u_output[0]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Threshold on P(class=0): if P(class=0) <= threshold, predict class=1
                predicted = (probs[:, 0] <= threshold).long()
                
                if return_probs:
                    all_probs.append(probs.detach().cpu().numpy())
                    all_targets.append(u_batch_y.cpu().numpy())
                
                # CrossEntropyLoss expects LONG targets, not float
                loss = self.loss_func(logits, u_batch_y.long())

                test_loss += loss.mean().item()
                total += u_batch_y.size(0)
                
                correct_preds = predicted.eq(u_batch_y).cpu().numpy()
                correct += np.sum(correct_preds)

                if separate: 
                    true_numpy = u_batch_y.cpu().numpy().squeeze()
                    # Positive class = 1, Negative class = 0 (inverted in data)
                    pos_idx = np.where(true_numpy == 0)[0]
                    neg_idx = np.where(true_numpy == 1)[0]

                    pos_correct += np.sum(correct_preds[pos_idx])
                    neg_correct += np.sum(correct_preds[neg_idx])

                    pos_total += len(pos_idx)
                    neg_total += len(neg_idx)
                
                if show_bar: 
                    progress_bar(batch_idx, len(u_validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        overall_acc = 100. * correct / total
        
        if return_probs:
            unlabeled_probs = np.concatenate(all_probs, axis=0)
            unlabeled_targets = np.concatenate(all_targets, axis=0)

            if unlabeled_targets.ndim > 1:
                unlabeled_targets = unlabeled_targets.squeeze()
            
            if not separate:
                return overall_acc, unlabeled_probs, unlabeled_targets
            else:
                pos_acc = 100. * pos_correct / pos_total if pos_total > 0 else 0
                neg_acc = 100. * neg_correct / neg_total if neg_total > 0 else 0
                return (overall_acc, pos_acc, neg_acc), unlabeled_probs, unlabeled_targets
        else:
            if not separate: 
                return overall_acc
            else: 
                pos_acc = 100. * pos_correct / pos_total if pos_total > 0 else 0
                neg_acc = 100. * neg_correct / neg_total if neg_total > 0 else 0
                return overall_acc, pos_acc, neg_acc

    def p_probs(self, p_loader): 
        self.eval()
        pp_probs = []
        with torch.no_grad():
            for batch_idx, p_data in enumerate(p_loader):
                p_batch_ids, p_batch_mask, p_batch_seg, p_batch_y, p_batch_pu_y, p_batch_len, p_batch_idxs = \
                    (item.to(self.device) for item in p_data)
                output = self.bert_model(p_batch_ids, p_batch_mask, p_batch_seg)
                probs = torch.nn.functional.softmax(output[0], dim=-1)[:, 0]  # P(class=0)
                pp_probs.append(probs.detach().cpu().numpy())
        
        pp_probs = np.concatenate(pp_probs, axis=0)
        if pp_probs.ndim > 1:
            pp_probs = pp_probs.squeeze()
        
        print(f"p_probs shape: {pp_probs.shape}")
        return pp_probs  

    def u_probs(self, u_loader):
        self.eval()
        pu_probs = None
        pu_targets = None
        with torch.no_grad():
            for batch_idx, u_data in enumerate(u_loader):
                u_batch_ids, u_batch_mask, u_batch_seg, u_batch_y, u_batch_pu_y, u_batch_len, u_batch_idxs = \
                    (item.to(self.device) for item in u_data)
                output = self.bert_model(u_batch_ids, u_batch_mask, u_batch_seg)
                    
                probs = torch.nn.functional.softmax(output[0], dim=-1) 
            
                if pu_probs is None: 
                    pu_probs = probs.detach().cpu().numpy().squeeze()
                    pu_targets = u_batch_y.cpu().numpy().squeeze()
                    
                else:
                    pu_probs = np.concatenate( (pu_probs, \
                        probs.detach().cpu().numpy().squeeze()))
                    pu_targets = np.concatenate( (pu_targets, \
                        u_batch_y.cpu().numpy().squeeze()))
                    
        
        return pu_probs, pu_targets

    def load_model(self, path):
        self.load_state_dict(
            state_dict=torch.load(path, map_location=self.device)
        )
        if torch.cuda.is_available(): self.cuda()