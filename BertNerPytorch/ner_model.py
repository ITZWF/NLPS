import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torch import nn
from pytorch_pretrained_bert import BertModel

START_TAG = 'START'
STOP_TAG = 'STOP'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BertNER(nn.Module):

    def __init__(self, embedding_dim=768, hidden_dim=100, char_lstm_dim=10, tag_to_ix=None):
        super(BertNER, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_channels = char_lstm_dim
        self.tag_to_ix = tag_to_ix

        self.bert = BertModel.from_pretrained('./BERT/').cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # self.softmax = nn.Softmax(dim=-1)
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.out_channels)

        self.transitions = nn.Parameter(torch.zeros(self.out_channels, self.out_channels))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def bert_emb(self, inputs):
        inputs = inputs.long()
        inputs_emb = torch.zeros(size=[inputs.size(0), inputs.size(1), self.embedding_dim]).cuda()
        with torch.no_grad():
            bert_layers = self.bert(inputs)[0]
            last_4_layers = bert_layers[-4:]
            for layer in last_4_layers:
                inputs_emb = torch.add(layer, inputs_emb)
        # [4, 60, 768] [batch_size, seq_len, bert_emb_dim]
        return inputs_emb.cuda()

    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        r = r.cuda()
        tags = tags.view(-1).long()
        score = torch.sum(feats[r, tags])

        return score

    def _get_lstm_features(self, inputs):
        embeds = self.bert_emb(inputs).cuda()
        # inputs [bs, seq_len] embeds [bs, sl, bert_dim]
        embeds = embeds.transpose(0, 1)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = lstm_out.reshape(-1, self.hidden_dim * 2)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.out_channels).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.out_channels).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = Variable(init_vvars)
        forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.out_channels, self.out_channels) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self._get_lstm_features(sentence)

        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        feats = self._get_lstm_features(sentence)
        # viterbi to get tag_seq
        score, tag_seq = self.viterbi_decode(feats)

        return score, tag_seq
