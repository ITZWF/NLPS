# encoding: utf-8
import pickle
from transformers import BertTokenizer

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag2ix = {"O": 0,
          "B-ORG": 1, "I-ORG": 2,
          "B-LOC": 3, "I-LOC": 4,
          START_TAG: 5, STOP_TAG: 6,
          }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    max_len = 0
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split('\t')
            sent_.append(char)
            tag_.append(label)
        else:
            if sent_ and tag_:
                if len(sent_) > max_len:
                    max_len = len(sent_)
                    if max_len > 500:
                        max_len = 500
                data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data, max_len


def data_encode(data, max_len):
    bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese/bert-base-chinese-vocab.txt')
    dataset = []
    for sent_, tag_ in data:
        sent_copy = ''.join(sent_)
        words = bert_tokenizer.tokenize(sent_copy)
        feature = bert_tokenizer.encode(words)
        tag_.insert(0, 'O')
        tag_.append('O')
        tag_copy = []
        for tg in tag_:
            tag_copy.append(tag2ix[tg])
        for _ in range(max_len - len(feature)):
            feature.append(bert_tokenizer.pad_token_id)
            tag_copy.append(tag2ix['O'])
        dataset.append((feature[:max_len], tag_copy[:max_len]))
    return dataset, max_len


def data_encode_limit(data, max_len=200):
    bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese/bert-base-chinese-vocab.txt')
    dataset = []
    num = 0
    for sent_, tag_ in data:
        feature = [bert_tokenizer.cls_token_id]
        for x in sent_:
            if x in bert_tokenizer.vocab:
                feature.append(bert_tokenizer.vocab[x])
            else:
                feature.append(bert_tokenizer.unk_token_id)
        feature.append(bert_tokenizer.sep_token_id)
        tag_.insert(0, 'O')
        tag_.append('O')
        tag_copy = []
        for tg in tag_:
            if tg in tag2ix:
                tag_copy.append(tag2ix[tg])
            else:
                tag_copy.append(tag2ix['O'])
        if len(feature) <= 200:
            # for _ in range(max_len - len(feature)):
            #     feature.append(bert_tokenizer.pad_token_id)
            #     tag_copy.append(tag2ix['O'])
            if len(feature) == len(tag_copy):
                dataset.append((feature[:max_len], tag_copy[:max_len]))
            else:
                num += 1
        else:
            num += 1
            print('ignore this sentence: %d' % num)
    return dataset, max_len


def pickle_data(data, outfile):
    outfile = open(outfile, mode='wb')
    pickle.dump(data, outfile)
    outfile.close()


if __name__ == '__main__':
    datas, _ = read_corpus('train_data')
    datas, _ = data_encode_limit(datas, 200)
    pickle_data(datas, 'train_data-org-loc.pkl')
    # datas, max_length = read_corpus('test_data')
    # datas, max_length = data_encode(datas, max_length)
    # pickle_data(datas, 'test_data.pkl')
