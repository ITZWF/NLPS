# coding=utf-8
from __future__ import print_function
from torch.utils.data import Dataset
from torch.utils import data
import torch
import pickle
import time

from transformers import BertTokenizer

# from utils import *
from ner_model import BertNER
from tqdm import tqdm


class NerDataset(Dataset):

    def __init__(self, ner_data):
        super(NerDataset, self).__init__()
        self.ner_data = ner_data
        self.train_data = list()
        for x, y in zip(ner_data['x_train'], ner_data['y_train']):
            self.train_data.append([x, y])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        [x, y] = self.train_data[item]
        return x, y


t = time.time()
models_path = "models/"

data1 = pickle.load(open('./dataset/dataMSRA.pkl', mode='rb'))
tokenizer = BertTokenizer.from_pretrained('./BERT/bert-base-chinese-vocab.txt')
word2id = tokenizer.vocab
train_data = NerDataset(data1)
train_lorder = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
tag_to_ix = data1['tag2id']

if __name__ == '__main__':
    model = BertNER(tag_to_ix=tag_to_ix)

    model.cuda()
    learning_rate = 0.0005
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = 0.0
    best_dev_F = -1.0
    best_test_F = -1.0
    best_train_F = -1.0
    all_F = [[0, 0, 0]]
    plot_every = 10
    eval_every = 20
    count = 0

    tag2id = {'B_ns': 0,
              'B_nr': 1,
              'B_nt': 2,
              'I_ns': 3,
              'I_nr': 4,
              'I_nt': 5,
              'O': 6,
              'START': 7,
              'STOP': 8}

    model.train(True)

    for epoch in range(1, 10001):
        pbar = tqdm(train_lorder)
        step_i = 0
        for td in pbar:
            tr = time.time()
            model.zero_grad()

            start_sent_in = torch.Tensor(td[0].size(0), 1).fill_(word2id['[CLS]'])
            end_sent_in = torch.Tensor(td[0].size(0), 1).fill_(word2id['[SEP]'])
            start_lb_in = torch.Tensor(td[1].size(0), 1).fill_(tag2id['START'])
            end_lb_in = torch.Tensor(td[1].size(0), 1).fill_(tag2id['STOP'])
            sentence_in = td[0]
            sentence_in = torch.cat([start_sent_in, sentence_in], dim=-1)
            sentence_in = torch.cat([sentence_in, end_sent_in], dim=-1)
            tags = td[1]
            tags = torch.cat([start_lb_in, tags], dim=-1)
            tags = torch.cat([tags, end_lb_in], dim=-1)

            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), tags.cuda())
            del sentence_in
            del tags

            loss += neg_log_likelihood
            neg_log_likelihood.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step_i += 1
            pbar.set_description('EPOCH: %d , loss: %.4f, ava_loss: %.4f' % (epoch, neg_log_likelihood, loss / step_i))

        torch.save(model.state_dict(), './models/ner-%d.pkl' % epoch)

        print(time.time() - t)
