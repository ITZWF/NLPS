# encoding: utf-8
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torchtext.data import Dataset, BucketIterator

from chatbot_model import ChatBot

data = pickle.load(open('./data/xhj.pkl', mode='rb'))
vocab = data['vocab']['src'].vocab
print(len(vocab))
fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}
pad = vocab.stoi['<PAD>']
train_dataset = Dataset(examples=data['train'], fields=fields)

train_iterator = BucketIterator(train_dataset, batch_size=20, device='cuda', train=True, shuffle=True)

if __name__ == '__main__':
    model = ChatBot(len(vocab), d_model=512, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048,
                    dropout=0.2, activation='relu')
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=4000, eta_min=0.00001, last_epoch=-1)
    epochs = 100
    for epoch in range(1, 1 + epochs):
        train_bar = tqdm(train_iterator)
        total_loss, total_acc, batch_num = 0, [], 0
        for ix, data in enumerate(train_bar):
            model.train()
            src, trg = data.src.long(), data.trg.long()
            src = src.cuda()
            trg = trg.cuda()
            predict = model(src, trg, pad_id=pad)    # shape [seq_len, bs, dim]
            shift_logits = predict[:-1, ..., :].contiguous()
            shift_labels = trg[1:, ...].contiguous()
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
            nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
            lr_schedule.step()
            total_loss += loss.item()
            batch_num += 1
            train_bar.set_description(
                f'[epoch: {epoch}] lr: {round(optimizer.param_groups[0]["lr"], 6)}, loss: {round(loss.item(), 4)}|'
                f'{round(total_loss / batch_num, 4)}, token acc: {round(accuracy.item() * 100, 4)} % |{round(np.mean(total_acc) * 100, 4)} %')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), './models/ChatBotModel-%d.pkl' % epoch)
