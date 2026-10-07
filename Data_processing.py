import os
import pickle
import random
import argparse

from tqdm import tqdm
from transformers import BertTokenizer

def FT_data(file, tokenizer=None, n_samples:int=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    if n_samples is not None:
        data = data[:n_samples]
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    cr = [ [sen for sen in a[1:]] for a in data]
    cr_list=[]
    cnt=0
    for s in tqdm(cr):
        s_list=[]
        for sen in s[:-1]:
            if len(sen)==0:
                cnt+=1
                continue
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+tokenizer.eos_token))
        s_list=s_list+[tokenizer.sep_token_id]
        s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)
    print(cnt)
    return y, cr_list

def create_PU_labels(y, r_PU, seed=42):
    """
    Create positive-unlabeled labels.
    All negatives (y=0) become unlabeled (-1).
    Exactly r_PU fraction of positives (y=1) become unlabeled (-1).
    
    :param y: original labels (0 or 1)
    :param r_PU: ratio of positive samples to mask as unlabeled (0.0 to 1.0)
    :param seed: random seed for reproducibility
    
    :return: PU labels (1 for labeled positive, -1 for unlabeled)
    """
    random.seed(seed)
    
    positive_indices = [i for i, label in enumerate(y) if label == 1]
    n_positives = len(positive_indices)
    n_to_mask = int(n_positives * r_PU)
    mask_indices = set(random.sample(positive_indices, n_to_mask))
    
    y_pu = []
    for i, label in enumerate(y):
        if label == 0 or i in mask_indices:
            y_pu.append(-1)
        else:
            y_pu.append(1)
    
    return y_pu

def PT_data(
        datadir:str,
        r_PU:float=None,
        y_indexes:list=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    if (r_PU is None) != (y_indexes is None):
        raise ValueError("r_PU and y_indexes must both be provided or both be None")

    data = open(f'{datadir}/train.txt', 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    cr = [[sen for sen in a[1:]] for a in data]

    crnew=[]
    if y_indexes is None:
        y = [int(a[0]) for a in data]
        for i,crsingle in enumerate(cr):
            if y[i]==1:
                crnew.append(crsingle)
    else:
        for i in y_indexes:
            crnew.append(cr[i])

    if r_PU is not None:
        pickle.dump(crnew, file=open(f'{datadir}/post_train_rPU{r_PU}.pkl', 'wb'))
    else:
        pickle.dump(crnew, file=open(f'{datadir}/post_train.pkl', 'wb'))

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['douban', 'ubuntu', 'e_commerce'], help='dataset name')
    parser.add_argument('--data_dir', type=str, default='data/', help='directory containing the dataset files')
    parser.add_argument('--n_samples', type=int, default=None, help='number of samples to process from the training data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    dataset = args.dataset
    datadir = os.path.join(args.data_dir, f'{dataset}_data')

    if dataset in ['douban', 'e_commerce']:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",do_lower_case=True)
    else:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)

    train, test, dev = {}, {}, {}
    dev_str = 'valid' if dataset=='ubuntu' else 'dev'
    train['y'], train['cr'] = FT_data(f'{datadir}/train.txt', tokenizer=bert_tokenizer, n_samples=args.n_samples)
    dev['y'], dev['cr'] = FT_data(f'{datadir}/{dev_str}.txt', tokenizer=bert_tokenizer, n_samples=args.n_samples)
    test['y'], test['cr']= FT_data(f'{datadir}/test.txt',tokenizer=bert_tokenizer, n_samples=args.n_samples)
    
    r_PUs = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
    with open(f'{datadir}/stats.txt', 'w') as stats_file:
        stats_msg = f"Original positives: {sum(1 for y in train['y'] if y == 1)}\n"
        stats_msg += f"Original negatives: {sum(1 for y in train['y'] if y == 0)}\n"
        for r_PU in r_PUs:
            train[f'y_pu_{r_PU}'] = create_PU_labels(train['y'], r_PU=r_PU)
            dev[f'y_pu_{r_PU}'] = create_PU_labels(dev['y'], r_PU=r_PU)
            test[f'y_pu_{r_PU}'] = create_PU_labels(test['y'], r_PU=r_PU)
            
            stats_msg += f"\nPU Label Statistics (r_PU={r_PU}):\n"
            stats_msg += f"Train Labeled positives: {sum(1 for y in train[f'y_pu_{r_PU}'] if y == 1)}\n"
            stats_msg += f"Train Unlabeled samples: {sum(1 for y in train[f'y_pu_{r_PU}'] if y == -1)}\n"
            stats_msg += f"Dev Labeled positives: {sum(1 for y in dev[f'y_pu_{r_PU}'] if y == 1)}\n"
            stats_msg += f"Dev Unlabeled samples: {sum(1 for y in dev[f'y_pu_{r_PU}'] if y == -1)}\n"
            stats_msg += f"Test Labeled positives: {sum(1 for y in test[f'y_pu_{r_PU}'] if y == 1)}\n"
            stats_msg += f"Test Unlabeled samples: {sum(1 for y in test[f'y_pu_{r_PU}'] if y == -1)}\n"

            y_indexes = [i for i, label in enumerate(train[f'y_pu_{r_PU}']) if label == 1]
            PT_data(datadir, r_PU=r_PU, y_indexes=y_indexes)
            
        print(stats_msg)
        stats_file.write(stats_msg)
    
    #posttraining data construction
    #does not include tokenization step
    PT_data(datadir, r_PU=None, y_indexes=None)

    data = train, dev, test
    pickle.dump(data, open(f'{datadir}/dataset_1M.pkl', 'wb'))

    print("Data processing completed.")