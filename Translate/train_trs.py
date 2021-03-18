# encoding: utf-8
import pickle

import torch
import numpy as np
from torchtext.data import Dataset, BucketIterator
from tqdm import tqdm

from model_trs import Trs
# from utils import LabelSmoothing
from utils import ScheduledOptim


def prepare_dataloaders(pkl, bs, device):
    batch_size = bs
    data = pickle.load(open(pkl, 'rb'))
    vocab_src = data['vocab']['src'].vocab
    vocab_tgt = data['vocab']['tgt'].vocab
    # trg_vocab_size = len(data['vocab']['trg'].vocab)

    fields = {'src': data['vocab']['src'], 'tgt': data['vocab']['tgt']}
    # fields = torchtext.data.Field(pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD, unk_token=UNK_WORD)
    train = Dataset(examples=data['train'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True, shuffle=True)

    return train_iterator, vocab_src, vocab_tgt


def train_model(epochs, train_iter, vocab_s, vocab_t, device=True):
    # unk = vocab.stoi['<UNK>']
    # sep = vocab.stoi['<SOS>']
    # cls = vocab.stoi['<EOS>']
    pad = vocab_t.stoi['<PAD>']
    model = Trs(vocab_size_src=len(vocab_s), vocab_size_tgt=len(vocab_t), dropout=0.3, d_model=256, nhead=4,
                num_decoder_layers=2, num_encoder_layers=2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=2)
    # lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
    lr_schedule = ScheduledOptim(optimizer, 2.0, 256, 2000)
    if device:
        model = model.cuda()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, total_acc, batch_num = 0, [], 0
        pbar = tqdm(train_iter, mininterval=2, leave=False)
        for idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            src, tgt = batch.src, batch.tgt
            if device:
                src, tgt = src.cuda(), tgt.cuda()
            pred = model(src=src, tgt=tgt, pad_id=pad)
            shift_logits = pred[:-1, ..., :].contiguous()
            shift_labels = tgt[1:, ...].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss.backward()
            _, preds = shift_logits.max(dim=-1)  # [S, B]
            not_ignore = shift_labels.ne(pad)
            num_targets = not_ignore.long().sum().item()  # the number of not pad tokens
            correct = (shift_labels == preds) & not_ignore
            correct = correct.float().sum()
            accuracy = correct / num_targets
            total_acc.append(accuracy.item())
            torch.nn.utils.clip_grad_value_(model.parameters(), 200.0)
            # optimizer.step()
            lr_schedule.step_and_update_lr()
            # self.warmup_scheduler.step()
            total_loss += loss.item()
            batch_num += 1
            pbar.set_description(
                f'[epoch: {epoch}] lr: {round(optimizer.param_groups[0]["lr"], 6)}, loss: {round(loss.item(), 4)}|'
                f'{round(total_loss / batch_num, 4)}, token acc: {round(accuracy.item() * 100, 4)} % |{round(np.mean(total_acc) * 100, 4)} %')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './modelsDir/TrsModel-%d.pkl' % epoch)


if __name__ == '__main__':
    batch_size = 24
    train_data, vocab_src, vocab_tgt = prepare_dataloaders('./data/zh-en.pkl', batch_size, 'cuda')
    train_model(2000, train_data, vocab_src, vocab_tgt, device=True)
