import pickle
import jieba
import torch
from torchtext.data import Dataset, BucketIterator

from model_trs import Trs


def prepare_dataloaders(pkl, bs, device):
    batch_size = bs
    data = pickle.load(open(pkl, 'rb'))
    # PAD_WORD = '<PAD>'
    # UNK_WORD = '<UNK>'
    # BOS_WORD = '<SOS>'
    # EOS_WORD = '<EOS>'
    # max_token_seq_len = 100
    # src_pad_idx = data['vocab']['src'].vocab.stoi['<PAD>']
    # trg_pad_idx = data['vocab']['trg'].vocab.stoi['<PAD>']
    #
    vocab_src = data['vocab']['src'].vocab
    vocab_tgt = data['vocab']['tgt'].vocab
    # trg_vocab_size = len(data['vocab']['trg'].vocab)

    fields = {'src': data['vocab']['src'], 'tgt': data['vocab']['tgt']}
    # fields = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD)
    train = Dataset(examples=data['train'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True, shuffle=True)

    return train_iterator, vocab_src, vocab_tgt


def model_predict(vocab_s, vocab_t):
    model = Trs(vocab_size_src=len(vocab_s), vocab_size_tgt=len(vocab_t), dropout=0.3, d_model=256, nhead=4,
                num_decoder_layers=2, num_encoder_layers=2)
    model.load_state_dict(torch.load('./modelsDir/TrsModel-10.pkl'))
    model.cuda()
    while True:
        text = input('输入对话: ')
        text_ls1 = jieba.lcut(text)
        PAD_WORD = '<PAD>'
        UNK_WORD = '<UNK>'
        BOS_WORD = '<SOS>'
        EOS_WORD = '<EOS>'
        text_ls = [vocab_s.stoi[BOS_WORD]]

        for t in text_ls1:
            if t in vocab_s.stoi:
                text_ls.append(vocab_s.stoi[t])
            else:
                text_ls.append(vocab_s.stoi[UNK_WORD])
        text_ls.append(vocab_s.stoi[EOS_WORD])
        if len(text_ls) <= 60:
            for _ in range(60-len(text_ls)):
                text_ls.append(vocab_s.stoi[PAD_WORD])
        text_ls = torch.LongTensor(text_ls).unsqueeze(0).transpose(0, 1).cuda()
        with torch.no_grad():
            predict = model.predict(text_ls, 20, vocab_t.stoi[PAD_WORD], vocab_t.stoi[BOS_WORD], vocab_t.stoi[EOS_WORD])
        predict = predict.cpu().numpy().tolist()
        predict_ls = []
        for p in predict:
            predict_ls.append(vocab_t.itos[p[0]])
        predicts = ' '.join(predict_ls)
        print(predicts)


if __name__ == '__main__':
    batch_size = 24
    train_data, vocab_src, vocab_tgt = prepare_dataloaders('./data/zh-en.pkl', batch_size, 'cuda')
    model_predict(vocab_src, vocab_tgt)
