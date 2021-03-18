# encoding: utf-8
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer

from utils import PositionEmbedding


class Trs(Transformer):

    def __init__(self, vocab_size_src, vocab_size_tgt, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder=None, custom_decoder=None):
        super(Trs, self).__init__()
        self.dropout = dropout
        self.position_encoder = PositionEmbedding(d_model=d_model, dropout=dropout, max_len=80)
        self.emb_src = nn.Embedding(vocab_size_src, d_model)
        self.emb_tgt = nn.Embedding(vocab_size_tgt, d_model)

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            from torch.nn import TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

        self.pred = nn.Linear(d_model, vocab_size_tgt)
        self._reset_parameters()

    # 获取padding mask
    @staticmethod
    def get_pad_mask(seq, pad_idx):
        seq = seq.transpose(0, 1)
        mask = seq == pad_idx
        return mask

    @staticmethod
    def generate_square_subsequent_mask(dim=512, sz: int = 10):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(dim, sz, sz)) == 1).transpose(1, 2)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, pad_id=0):
        emb_src = self.emb_src(src)
        emb_tgt = self.emb_tgt(tgt)
        src_emb = self.position_encoder(emb_src)
        tgt_emb = self.position_encoder(emb_tgt)
        src_pad_mask = self.get_pad_mask(src, pad_id)
        tgt_pad_mask = self.get_pad_mask(tgt, pad_id)
        tgt_mask = self.generate_square_subsequent_mask(src.size(1) * self.nhead, tgt.size(0))  # heads=8
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        pred = F.relu(self.pred(output))
        return pred

    @torch.no_grad()
    def predict(self, src, max_len, pad_id, cls_id, sep_id):
        emb_src = self.emb_src(src)
        src_emb = self.position_encoder(emb_src)
        src_pad_mask = self.get_pad_mask(src, pad_id)
        batch_size = src.size(1)
        stop_flag = [False] * batch_size
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        tgt = torch.LongTensor([cls_id] * batch_size).unsqueeze(0).cuda()  # [1, B]
        for idx in range(1, max_len + 1):
            bert_tgt = self.emb_tgt(tgt)
            tgt_emb = self.position_encoder(bert_tgt)
            tgt_mask = self.generate_square_subsequent_mask(src.size(1) * self.nhead, idx)
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # [1, B, E]
            pred = self.pred(output[-1, :, :])  # [B, V]
            next_token = torch.argmax(pred, -1).view(-1, 1)
            tgt = torch.cat([tgt, next_token], dim=0)  # [S+1, B]
            for idx_, token_i in enumerate(next_token.squeeze(0)):
                if token_i == sep_id:
                    stop_flag[idx_] = True
            if sum(stop_flag) == len(stop_flag):
                break
        return tgt
