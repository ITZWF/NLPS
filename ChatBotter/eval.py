import pickle
import jieba
import torch
from torchtext.data import Dataset, BucketIterator

from chatbot_model import ChatBot


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
    vocab = data['vocab']['src'].vocab
    # trg_vocab_size = len(data['vocab']['trg'].vocab)

    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}
    # fields = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD)
    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True, shuffle=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator, vocab


def model_predict(vocab):
    model = ChatBot(len(vocab), d_model=512, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048,
                    dropout=0.2, activation='relu')
    model.load_state_dict(torch.load('./models/ChatBotQy-360.pkl'))
    model.cuda()
    while True:
        text = input('输入对话: ')
        text_ls1 = jieba.lcut(text)
        PAD_WORD = '<PAD>'
        UNK_WORD = '<UNK>'
        BOS_WORD = '<SOS>'
        EOS_WORD = '<EOS>'
        text_ls = [vocab.stoi[BOS_WORD]]

        for t in text_ls1:
            if t in vocab.stoi:
                text_ls.append(vocab.stoi[t])
            else:
                text_ls.append(vocab.stoi[UNK_WORD])
        text_ls.append(vocab.stoi[EOS_WORD])
        if len(text_ls) <= 40:
            for _ in range(40-len(text_ls)):
                text_ls.append(vocab.stoi[PAD_WORD])
        text_ls = torch.LongTensor(text_ls).unsqueeze(0).transpose(0, 1).cuda()
        with torch.no_grad():
            predict = model.predict(text_ls, 40, vocab.stoi[PAD_WORD], vocab.stoi[BOS_WORD], vocab.stoi[EOS_WORD])
        predict = predict.cpu().numpy().tolist()
        predict_ls = []
        for p in predict:
            predict_ls.append(vocab.itos[p[0]])
        predicts = ' '.join(predict_ls)
        print(predicts)


if __name__ == '__main__':
    _, _, vocabs = prepare_dataloaders('./data/xhj.pkl', 20, 'cuda')
    model_predict(vocabs)