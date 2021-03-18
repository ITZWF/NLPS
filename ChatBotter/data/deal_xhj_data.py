# encoding: utf-8
import re
import jieba
import pickle
import torchtext


def deal_data(infile, src_tgt):
    infile = open(infile, mode='r', encoding='utf-8')
    src_tgt = open(src_tgt, mode='w', encoding='utf-8')

    content = []

    while True:
        lines = infile.readlines(4096)
        if not lines:
            break
        for line in lines:
            if line.strip().startswith('E'):
                if len(content) == 2:
                    [src_lan, tgt_lan] = content
                    content = []
                    src_lan = re.sub(r'^M ', '', src_lan)
                    tgt_lan = re.sub(r'^M ', '', tgt_lan)
                    src_lan = ' '.join(jieba.lcut(src_lan))
                    tgt_lan = ' '.join(jieba.lcut(tgt_lan))
                    src_lan = re.sub(r'[\t ]+', ' ', src_lan)
                    tgt_lan = re.sub(r'[\t ]+', ' ', tgt_lan)
                    src_lan = src_lan.strip()
                    tgt_lan = tgt_lan.strip()
                    if src_lan and tgt_lan:
                        src_tgt.write(src_lan + '\t' + tgt_lan + '\n')
                else:
                    content = []
            elif line.strip().startswith('M'):
                content.append(line)
    infile.close()
    src_tgt.close()


def filter_examples_with_length(x):
    return len(vars(x)['src']) <= 40 and len(vars(x)['trg']) <= 40


def pickle_data(src_tgt, pkl):
    PAD_WORD = '<PAD>'
    UNK_WORD = '<UNK>'
    BOS_WORD = '<SOS>'
    EOS_WORD = '<EOS>'
    src_field = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD)
    tgt_field = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD)
    train = torchtext.data.TabularDataset.splits(path='./', train=src_tgt, format='tsv',
                                                      fields=[('src', src_field), ('trg', tgt_field)],
                                                      filter_pred=filter_examples_with_length)
    src_field.build_vocab(train[0].src, min_freq=10)
    tgt_field.build_vocab(train[0].trg, min_freq=10)
    # 合并字典
    for w, _ in src_field.vocab.stoi.items():
        if w not in tgt_field.vocab.stoi:
            tgt_field.vocab.stoi[w] = len(tgt_field.vocab.stoi)
    tgt_field.vocab.itos = [None] * len(tgt_field.vocab.stoi)
    for w, i in tgt_field.vocab.stoi.items():
        tgt_field.vocab.itos[i] = w
    src_field.vocab.stoi = tgt_field.vocab.stoi
    src_field.vocab.itos = tgt_field.vocab.itos

    # for ii, xs in enumerate(train.examples):
    #     s, t = xs['src'], xs['trg']
    #     for i1, s1 in enumerate(s):
    #         if s1 not in src_field.vocab.stoi:
    #             s[i1] = UNK_WORD
    #     for i1, s1 in enumerate(t):
    #         if s1 not in src_field.vocab.stoi:
    #             t[i1] = UNK_WORD
    #     train.examples[ii]['src'] = s
    #     train.examples[ii]['trg'] = t

    data = {'vocab': {'src': src_field, 'trg': src_field}, 'train': train[0].examples}
    pickle.dump(data, open(pkl, mode='wb'))


if __name__ == '__main__':
    # deal_data('xiaohuangji50w_nofenci.conv', 'xhj.csv')
    pickle_data('./xhj.csv', './xhj.pkl')
