# coding=utf-8
from __future__ import print_function
import os

import re
from copy import copy

import torch
from transformers import BertTokenizer

from bert_crf_parallel import BertCrfParallel

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag2ix = {"O": 0,
          "B-ORG": 1, "I-ORG": 2,
          "B-LOC": 3, "I-LOC": 4,
          START_TAG: 5, STOP_TAG: 6,
          }

id2tag = {}
for k, v in tag2ix.items():
    id2tag[v] = k

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/bert-base-chinese-vocab.txt')


def eval_(model_):
    while True:
        text = input('输入： ')
        if text in ['q', 'quit', 'exit']:
            break
        per = []
        org = []
        loc = []
        # per_ls = []
        org_ls = []
        loc_ls = []
        sentences = cut_sentences(text)
        if sentences:
            for text in sentences:
                sent_ = [i for i in text]
                sent_copy = copy(sent_)
                sent_copy.insert(0, '[CLS]')
                sent_copy.append('[SEP]')
                feature = [tokenizer.cls_token_id]
                for x in sent_:
                    if x in tokenizer.vocab:
                        feature.append(tokenizer.vocab[x])
                    else:
                        feature.append(tokenizer.unk_token_id)
                feature.append(tokenizer.sep_token_id)
                if len(feature) > 500:
                    feature = feature[:500]
                tokens = torch.tensor(feature).long().view(1, -1).cuda()
                val, out = model_(tokens)
                pred = []
                for x in out:
                    pred.append(id2tag[x])
                for i, pre in enumerate(pred):
                    try:
                        if '-' in pre:
                            if pre.startswith('B'):
                                if pre.endswith('PER'):
                                    per.append([sent_copy[i]])
                                elif pre.endswith('ORG'):
                                    org.append([sent_copy[i]])
                                elif pre.endswith('LOC'):
                                    loc.append([sent_copy[i]])
                            elif pre.startswith('I'):
                                if pre.endswith('PER'):
                                    per[-1].append(sent_copy[i])
                                elif pre.endswith('ORG'):
                                    org[-1].append(sent_copy[i])
                                elif pre.endswith('LOC'):
                                    loc[-1].append(sent_copy[i])
                    except Exception as e:
                        print(e.__str__())
        # for n in per:
        #     per_ls.append(''.join(n))
        for n in org:
            new_n = re.sub(r'\[UNK]|\[SEP]|##', '', ''.join(n))
            if len(new_n) >= 3:
                org_ls.append(new_n)
        for n in loc:
            new_n = re.sub(r'\[UNK]|\[SEP]|##', '', ''.join(n))
            if len(new_n) >= 3:
                loc_ls.append(new_n)
        print('LOC: %s' % str(loc_ls))
        print('ORG: %s' % str(org_ls))


def cut_sentences(content):
    # 结束符号，包含中文和英文的
    end_flag = ['?', '!', '？', '！', '。', '…', ',', '，', '；']
    special_flag = ['一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、',
                    '1、', '2、', '3、', '4、', '5、', '6、', '7、', '8、', '9、']

    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        # 拼接字符
        tmp_char += char

        # 判断是否已经到了最后一位
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break

        # 判断此字符是否为结束符号
        if char in end_flag:
            # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                if tmp_char.strip():
                    if re.findall(r'[\u4e00-\u9fa5]', tmp_char.strip()):
                        sentences.append(tmp_char.strip())
                tmp_char = ''
    sentences_new = []
    forward_index = 0
    for sent in sentences:
        if re.findall(r'一、|二、|三、|四、|五、|六、|七、|八、|九、|十、|\d+、', sent):
            for char in special_flag:
                if char in sent:
                    after_index = sent.index(char)
                    if abs(after_index - forward_index) > 4:
                        sentences_new.append(sent[forward_index: after_index])
                    forward_index = after_index
        else:
            if len(sent) > 4:
                sentences_new.append(sent)

    return sentences_new


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = BertCrfParallel(tag_to_ix=tag2ix, cuda_flag=True)
    model = model.to('cuda:0')
    model.load_state_dict(torch.load('./models/org-loc-ner-46.pt'))
    model.eval()
    eval_(model)
