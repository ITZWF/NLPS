# encoding: utf-8
import re
import jieba
import pickle
import torchtext
import nltk


def deal_data(in_src, in_tgt, outfile):
    in_src = open(in_src, mode='r', encoding='utf-8')
    in_tgt = open(in_tgt, mode='r', encoding='utf-8')
    outfile = open(outfile, mode='w', encoding='utf-8')

    while True:
        line1 = in_src.readline()
        line2 = in_tgt.readline()
        if not line1 and not line2:
            break
        src_lan = ' '.join(jieba.lcut(line1.strip()))
        en_token = nltk.word_tokenize(line2.strip(), language='english')
        tgt_lan = ' '.join(en_token)
        src_lan = re.sub(r'[\t ]+', ' ', src_lan)
        tgt_lan = re.sub(r'[\t ]+', ' ', tgt_lan)
        src_lan = src_lan.strip()
        tgt_lan = tgt_lan.strip()
        if src_lan and tgt_lan:
            outfile.write(src_lan + '\t' + tgt_lan + '\n')
    in_src.close()
    in_tgt.close()
    outfile.close()


def filter_examples_with_length(x):
    return len(vars(x)['src']) <= 60 and len(vars(x)['tgt']) <= 60


def pickle_qy_data(src_tgt, pkl):
    PAD_WORD = '<PAD>'
    UNK_WORD = '<UNK>'
    BOS_WORD = '<SOS>'
    EOS_WORD = '<EOS>'
    src_field = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD, tokenizer_language='zh')
    tgt_field = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD, tokenizer_language='en', lower=True)
    train = torchtext.data.TabularDataset.splits(path='./', train=src_tgt, format='tsv',
                                                      fields=[('src', src_field), ('tgt', tgt_field)],
                                                      filter_pred=filter_examples_with_length)
    src_field.build_vocab(train[0].src, max_size=8000)
    tgt_field.build_vocab(train[0].tgt, max_size=8000)
    # # 合并字典
    # for w, _ in src_field.vocab.stoi.items():
    #     if w not in tgt_field.vocab.stoi:
    #         tgt_field.vocab.stoi[w] = len(tgt_field.vocab.stoi)
    # tgt_field.vocab.itos = [None] * len(tgt_field.vocab.stoi)
    # for w, i in tgt_field.vocab.stoi.items():
    #     tgt_field.vocab.itos[i] = w
    # src_field.vocab.stoi = tgt_field.vocab.stoi
    # src_field.vocab.itos = tgt_field.vocab.itos

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

    data = {'vocab': {'src': src_field, 'tgt': tgt_field}, 'train': train[0].examples}
    pickle.dump(data, open(pkl, mode='wb'))


if __name__ == '__main__':
    # deal_data('news-commentary-v13.zh-en.zh', 'news-commentary-v13.zh-en.en', 'zh-en.csv')
    pickle_qy_data('zh-en.csv', 'zh-en.pkl')
