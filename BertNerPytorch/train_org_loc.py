# encoding: utf-8
import os
import torch
# from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from bert_crf_parallel import BertCrfParallel
from utils import CrfDataset, pad_collect_fn

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag2ix = {"O": 0,
          "B-ORG": 1, "I-ORG": 2,
          "B-LOC": 3, "I-LOC": 4,
          START_TAG: 5, STOP_TAG: 6,
          }

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/bert-base-chinese-vocab.txt')
train_dataset = CrfDataset('./data/train_data-org-loc.pkl')
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, collate_fn=pad_collect_fn)
# train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
log_file = open('./data/train-data-org-loc.log', mode='w', encoding='utf-8')

if __name__ == '__main__':
    cuda_flag = True
    if cuda_flag:
        model = BertCrfParallel(tag2ix, cuda_flag=cuda_flag)
        # model = nn.DataParallel(model, device_ids=[0])
        model = model.to('cuda:0')
    else:
        model = BertCrfParallel(tag2ix, cuda_flag=cuda_flag)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.003)
    for epoch in range(100):
        losses = 0
        pbar = tqdm(train_loader)
        iix = 0
        for ix, data in enumerate(pbar):
            src, trg = data
            if cuda_flag:
                src = src.cuda()
                trg = trg.cuda()
            # src = pad_sequence(src, padding_value=bert_tokenizer.pad_token_id)
            # trg = pad_sequence(trg, padding_value=tag2ix['O'])
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            loss = model.neg_log_likelihood_parallel(src, trg)
            losses += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            pbar.set_description('Epoch %d, ava_loss: %.4f, loss: %.4f' % (epoch + 1, losses / (ix + 1), loss))
            iix += 1
            log_file.write('Epoch %d, ava_loss: %.4f, loss: %.4f' % (epoch + 1, losses / (ix + 1), loss))
        print('Epoch %d, ava_loss: %.4f' % (epoch + 1, losses / iix))
        log_file.write('Epoch %d Result, ava_loss: %.4f' % (epoch + 1, losses / iix))
        model.eval()
        torch.save(model.state_dict(), './models/org-loc-ner-%d.pt' % (epoch + 1))
    log_file.close()
