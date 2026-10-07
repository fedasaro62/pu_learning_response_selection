# Data

The corpora are not redistributed here. Download them and place the raw files as
follows, then run `Data_processing.py` to build the tokenized pickles and the
positive-unlabeled label sets.

## Ubuntu Corpus V1

Preprocessed version of Xu et al. (2017), as released by BERT-FP:
https://github.com/hanjanghoon/BERT_FP

```
data/ubuntu_data/train.txt
data/ubuntu_data/valid.txt
data/ubuntu_data/test.txt
```

## E-Commerce Corpus

Zhang et al. (2018), as released by BERT-FP:
https://github.com/hanjanghoon/BERT_FP

```
data/e_commerce_data/train.txt
data/e_commerce_data/dev.txt
data/e_commerce_data/test.txt
```

## Format

One example per line, tab-separated: a binary label, then the context
utterances, then the candidate response.

```
1<TAB>utterance_1<TAB>utterance_2<TAB>...<TAB>response
```

## Build

```bash
python Data_processing.py ubuntu
python Data_processing.py e_commerce
```

This writes, per corpus:

- `dataset_1M.pkl` — tokenized `(train, dev, test)` splits, each carrying the
  true labels `y` and the PU labels `y_pu_<r_PU>` for several PU ratios.
- `post_train.pkl` — all positive context-response pairs, for fully supervised
  post-training.
- `post_train_rPU<r>.pkl` — only the labeled positives, for PU post-training.
- `stats.txt` — label counts per PU ratio.
