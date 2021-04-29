import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from src.module import *

class Transformer(nn.Module):
    def __init__(self, config, args):
        super(Transformer, self).__init__()
        self.config = config
        self.dataset = args.dataset
        self.enc_voc = config.data_info[self.dataset].enc_voc
        self.dec_voc = config.data_info[self.dataset].dec_voc
        self.h_units = config.model.h_units
        self.d_rate = config.model.d_rate
        self.n_blocks = config.model.n_blocks
        self.n_head = config.model.n_head
        #encoder
        self.enc_emb = embedding(self.enc_voc,self.h_units, scale=True)
        self.enc_positional_encoding = positional_encoding(self.h_units, False, False)
        self.enc_dropout = nn.Dropout(self.d_rate)
        for i in range(self.n_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(self.h_units, self.n_head, self.d_rate, casuality=False))
            self.__setattr__('enc_feed_forward_%d'%i, feedforward(self.h_units, [4*self.h_units, self.h_units]))

        #decoder
        self.dec_emb = embedding(self.dec_voc,self.h_units, scale=True)
        self.dec_positional_encoding = positional_encoding(self.h_units, False, False)
        self.dec_dropout = nn.Dropout(self.d_rate)
        for i in range(self.n_blocks):
            self.__setattr__("dec_self_attention_%d"%i, multihead_attention(self.h_units, self.n_head, self.d_rate, casuality=True))
            self.__setattr__("dec_vanilla_attention_%d"%i, multihead_attention(self.h_units, self.n_head, self.d_rate, casuality=False))
            self.__setattr__("dec_feed_forward_%d"%i, feedforward(self.h_units, [4*self.h_units, self.h_units]))

        self.logits_layer = nn.Linear(self.h_units, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x, y):
        self.decoder_inputs = torch.cat([torch.ones(y[:, :1].size()).long(), y[:, :-1]], dim=-1) # <s> : id =1

        # encoder
        self.enc = self.enc_emb(x)
        self.enc += self.enc_positional_encoding(x)
        self.enc = self.enc_dropout(self.enc)

        for i in range(self.n_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d'%i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d'%i)(self.enc)

        # decoder
        self.dec = self.dec_emb(self.decoder_inputs)
        self.dec += self.dec_positional_encoding(self.decoder_inputs)
        self.dec = self.dec_dropout(self.dec)

        for i in range(self.n_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d'%i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d'%i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d'%i)(self.dec)

        self.logits = self.logits_layer(self.dec)
        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.dec_voc)
        _, self.preds = torch.max(self.logits, -1)
        self.istarget = (1.-y.eq(0.).float()).view(-1) # 0 : unkown 제거
        self.acc = torch.sum(self.preds.eq(y).float().view(-1)*self.istarget) / torch.sum(self.istarget)

        #loss

        self.logits = torch.flatten(self.logits,0,1)
        y = torch.flatten(y, 0, 1)
        loss = F.cross_entropy(self.logits, y)
        print(loss.item())
        return loss, self.acc



