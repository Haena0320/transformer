import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from src.module import *


class Transformer(nn.Module):
    def __init__(self, config, args):
        super(Transformer, self).__init__()
        self.config = config
        self.dataset = args.dataset
        if self.dataset =="en_de":
            self.enc_voc = config.data_info[self.dataset].vocab_size
            self.dec_voc = config.data_info[self.dataset].vocab_size
        self.h_units = config.model.h_units
        self.d_rate = config.model.d_rate
        self.n_blocks = config.model.n_blocks
        self.n_head = config.model.n_head
        #encoder
        self.enc_emb = embedding(self.enc_voc,self.h_units)
        self.enc_positional_encoding = positional_encoding(self.h_units, True)
        self.enc_dropout = nn.Dropout(self.d_rate)

        for i in range(self.n_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(self.h_units, self.n_head, self.d_rate, False))
            self.__setattr__('enc_feed_forward_%d'%i, feedforward(self.h_units, [4*self.h_units, self.h_units]))

        #decoder
        self.dec_emb = embedding(self.dec_voc,self.h_units, scale=True)
        self.dec_positional_encoding = positional_encoding(self.h_units, True)
        self.dec_dropout = nn.Dropout(self.d_rate)
        for i in range(self.n_blocks):
            self.__setattr__("dec_self_attention_%d"%i, multihead_attention(self.h_units, self.n_head, self.d_rate, True))
            self.__setattr__("dec_vanilla_attention_%d"%i, multihead_attention(self.h_units, self.n_head, self.d_rate, False))
            self.__setattr__("dec_feed_forward_%d"%i, feedforward(self.h_units, [4*self.h_units, self.h_units]))

        #predict next word
        self.logits_layer = nn.Linear(self.h_units, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x, y):
        print(y)
        self.decoder_inputs = torch.cat([torch.ones(y[:, :1].size()).long(), y[:, :-1]], dim=-1) # <s> : id =1
        print(self.decoder_inputs)

        # encoder
        self.enc = self.enc_emb.forward(x)
        self.enc += self.enc_positional_encoding.forward(x)
        self.enc = self.enc_dropout(self.enc)

        for i in range(self.n_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d'%i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d'%i)(self.enc)

        # decoder
        self.dec = self.dec_emb.forward(self.decoder_inputs)
        self.dec += self.dec_positional_encoding.forward(self.decoder_inputs)
        self.dec = self.dec_dropout(self.dec)

        for i in range(self.n_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d'%i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d'%i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d'%i)(self.dec)

        # predict next word
        self.logits = self.logits_layer(self.dec) # self.logits : (bs, sequence_length, vocab_size)
        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.dec_voc) # self.probs :  (bs, sequence_length, vocab_size)

        _, self.preds = torch.max(self.logits, -1)
        self.istarget = (1.-y.eq(0.).float()).view(-1) # 0 : zero padding 제거
        self.acc = torch.sum(self.preds.eq(y).float().view(-1)*self.istarget) / torch.sum(self.istarget)

        #loss
        self.logits = torch.flatten(self.logits,0,1)
        y = torch.flatten(y, 0, 1)
        loss = F.cross_entropy(self.logits, y)
        print(loss.item())
        print(self.acc)
        return loss, self.acc



