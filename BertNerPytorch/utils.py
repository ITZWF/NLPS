# encoding: utf-8
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/bert-base-chinese-vocab.txt')
START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag2ix = {"O": 0,
          "B-ORG": 1, "I-ORG": 2,
          START_TAG: 3, STOP_TAG: 4,
          }


def load_data(infile):
    data = pickle.load(open(infile, mode='rb'))
    return data


class CrfDataset(Dataset):

    def __init__(self, infile):
        super(CrfDataset, self).__init__()
        self.data = load_data(infile)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src = torch.tensor(src).long()
        trg = torch.tensor(trg).long()
        return src, trg

    def __len__(self):
        return len(self.data)


def pad_collect_fn(data):
    batch_size = len(data)
    max_len = max([len(i[0]) for i in data])
    new_src = torch.zeros([batch_size, max_len])
    new_trg = torch.zeros([batch_size, max_len])
    for ix, st in enumerate(data):
        src, trg = st
        if len(src) < max_len:
            src_fill = torch.tensor([bert_tokenizer.pad_token_id] * (max_len - len(src)))
            trg_fill = torch.tensor([tag2ix['O']] * (max_len - len(trg)))
            src = torch.cat([src, src_fill], dim=-1)
            trg = torch.cat([trg, trg_fill], dim=-1)
        new_src[ix] = src
        new_trg[ix] = trg
    return new_src.long(), new_trg.long()
