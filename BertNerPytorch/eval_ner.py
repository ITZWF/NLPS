# coding=utf-8
from __future__ import print_function
import pickle

import torch
import time
# from torch.autograd import Variable
from transformers import BertTokenizer

from ner_model import BertNER

data = pickle.load(open('./dataset/dataMSRA.pkl', mode='rb'))
t = time.time()

id2tag = data['id2tag']
model = BertNER(tag_to_ix=data['tag2id'])
del data
model.load_state_dict(torch.load('./models/ner-11.pkl'))
tokenizer = BertTokenizer.from_pretrained('./BERT/bert-base-chinese-vocab.txt')

model.cuda()
model.eval()


def eval_(model_):
    while True:
        text = input('输入： ')
        tokens = tokenizer.encode(text)
        words = tokenizer.convert_ids_to_tokens(tokens)
        tokens = torch.LongTensor(tokens).view(1, -1).cuda()
        val, out = model_(tokens)
        # print(tokens)
        # print(val)
        # print(out)
        pred = []
        for x in out:
            pred.append(id2tag[x])
        # print(words)
        # print(pred)
        nr = []
        nt = []
        ns = []
        nrs = []
        nts = []
        nss = []
        for i, pre in enumerate(pred):
            if '_n' in pre:
                if pre.startswith('B'):
                    if pre.endswith('r'):
                        nr.append([words[i]])
                    elif pre.endswith('t'):
                        nt.append([words[i]])
                    elif pre.endswith('s'):
                        ns.append([words[i]])
                elif pre.startswith('I'):
                    if pre.endswith('r'):
                        nr[-1].append(words[i])
                    elif pre.endswith('t'):
                        nt[-1].append(words[i])
                    elif pre.endswith('s'):
                        ns[-1].append(words[i])
        for n in nr:
            nrs.append(''.join(n))
        for n in ns:
            nss.append(''.join(n))
        for n in nt:
            nts.append(''.join(n))
        print('人物： ', nrs)
        print('组织机构： ', nts)
        print('地点： ', nss)


if __name__ == '__main__':
    eval_(model)

# for l in range(1, 6):
#     print('maxl=', l)
#     eval(model, test_data, l)
#     # print()
# # for i in range(10):
# #     eval(model, test_data, 100)

