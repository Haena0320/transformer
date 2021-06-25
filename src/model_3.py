import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm
import math
import numpy as np

class WordEncoding(nn.Module):
    def __init__(self, embed_weights, d_model):
        super(WordEncoding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_weights, freeze=False, padding_idx=0)
        self.d_model = d_model

    def forward(self,x):
        return self.embedding(x)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, device=None):
        super(PositionEncoding, self).__init__()
        self.device=device
        self.position_emb = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * - (math.log(1e4) / d_model))
        self.position_emb[:, 0::2] = torch.sin(position*div_term)
        self.position_emb[:, 1::2] = torch.cos(position*div_term)

    def forward(self, inputs):
        # batch size, sequence
        outputs = self.position_emb[:inputs.size(1), :]
        outputs = outputs.unsqueeze(0).repeat(inputs.size(0), 1, 1).to(self.device)
        return outputs
    
class Embedding(nn.Module):
    def __init__(self,d_model, embed_weights, device):
        super(Embedding, self).__init__()
        self.word_embed = WordEncoding(embed_weights, d_model).to(device)
        self.posit_embed = PositionEncoding(d_model=d_model, device=device)
        self.d_model = d_model
        #self.norm = LayerNorm(d_model)
        self.dropout = Dropout(p=0.1)

    def forward(self, input):
        output = self.word_embed(input)
        output *= self.d_model**(0.5)
        output += self.posit_embed(input)
        return self.dropout(output)
    
    
class ScaledDotProduct(nn.Module):
    def __init__(self, ahead_mask=None,dropout=0.1, device=None):
        super(ScaledDotProduct, self).__init__()
        self.ahead_mask = ahead_mask
        self.dropout = dropout
        self.device = device
    def forward(self, q, k, v, zero_mask=None):
        """
        
        :param q: (bs, h, seq_q, d_q)
        :param k: (bs, h, seq_k, d_k)
        :param v: (bs, h, seq_v, d_v)
        :param zero_mask: (bs, 1, 1, seq) or (bs, 1, seq, seq)
        :return: (bs, h, seq_q, d_v)
        """
        d_k = k.size(-1)
        att = torch.matmul(q, k.transpose(2,3)).to(self.device) #(bs, h, seq_q, seq_k)
        att = att / d_k**0.5

        if zero_mask is not None:
            att += zero_mask *(-1e+9)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att)
        att = torch.matmul(att, v)
        return att


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout, device, mask):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d = d_model // h
        assert d_model == self.h * self.d
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProduct(mask, device)
        self.linear_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout=nn.Dropout()
        self.layer_norm = nn.LayerNorm(d_model)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_o.weight)

    def forward(self, q, k, v, zero_mask):
        """
        :param q: (bs, seq_q, embed)
        :param k: (bs, seq_k, embed)
        :param v: (bs, seq_v, embed)
        :param zero_mask: (bs, 1, 1, seq) or (bs, 1, seq, seq)
        :return: (bs, seq_q, d_model)
        """
        bs, seq, _= q.size()
        residual = q.clone()
        _, seq_k, _ = k.size()

        q = self.linear_q(q).view(bs, seq, self.h, self.d).transpose(1,2)
        k = self.linear_k(k).view(bs, seq_k, self.h, self.d).transpose(1,2)
        v = self.linear_v(v).view(bs, seq_k, self.h, self.d).transpose(1,2)

        att = self.attention(q, k, v, zero_mask)
        att = att.transpose(1,2) 
        att = att.contiguous().view(bs, seq, -1)
        out = self.linear_o(att)
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        return out
        
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout()
        self.layer_norm = nn.LayerNorm(d_model)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        residual = x.clone()
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        return out

class Encoder_Sublayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout, device):
        super(Encoder_Sublayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, h, dropout, device, mask=False)
        self.position_feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

    def init_weights(self):
        self.multi_head_attention.init_weights()
        self.position_feed_forward.init_weights()

    def forward(self, x, enc_pad_mask):
        att = self.multi_head_attention(x,x,x,enc_pad_mask)
        out = self.position_feed_forward(att)
        return out

class Decoder_Sublayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout, device):
        super(Decoder_Sublayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, h, dropout, device, mask=False)
        self.multi_head_attention = MultiHeadAttention(d_model, h, dropout, device, mask=True)
        self.position_feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

    def init_weights(self):
        self.multi_head_attention.init_weights()
        self.multi_head_attention.init_weights()
        self.position_feed_forward.init_weights()

    def forward(self,x,hs, dec_combined_mask, dec_pad_mask):
        att1 = self.masked_multi_head_attention(x ,x, x, dec_combined_mask)
        att2 = self.multi_head_attention(att1, hs, hs, dec_pad_mask)
        out = self.position_feed_forward(att2)
        return out
    
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout, num_layers, device):
        super(Encoder, self).__init__()
        self.layers = ModuleList()
        for i  in range(num_layers):
            self.layers.append(Encoder_Sublayer(d_model, d_ff ,h, dropout, device))
        self.layers.to(device)
        self.num_layers = num_layers

    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()
            
    def forward(self, src, enc_pad_mask):
        for layer in self.layers:
            src = layer(src, enc_pad_mask)
        return src

class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout, num_layers, device):
        super(Decoder, self).__init__()
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(Decoder_Sublayer(d_model, d_ff, h, dropout, device))
        self.layers.to(device)
        self.num_layers = num_layers

    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()

    def forward(self, src,hs,dec_combined_mask, dec_pad_mask):
        for layer in self.layers:
            src = layer(src,hs ,dec_combined_mask, dec_pad_mask)
        return src


class TransformerModel(nn.Module):
    def __init__(self, vocab, D, dropout, num_layers, d_model, d_ff, h, device):
        super(TransformerModel, self).__init__()
        self.device = device
        
        self.emb_weights = nn.parameter.Parameter(torch.empty(vocab, d_model), requires_grad=True)
        nn.init.normal_(self.emb_weights, mean=0, std=d_model**(-0.5))
        
        self.embedding = Embedding(D, self.emb_weights, device)
        #encoder = Encoder_Sublayer(d_model, d_ff, h, dropout, device)
        #decoder = Decoder_Sublayer(d_model, d_ff, h, dropout, device)
        self.encoder = Encoder(d_model, d_ff, h, dropout, num_layers, device)
        self.decoder = Decoder(d_model, d_ff, h, dropout, num_layers, device)
        self.fc = nn.Linear(d_model, vocab, bias=False)

    def init_weights(self):
        print("model initialized .. ")
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.fc.weight_ = self.emb_weights.T

    def forward(self, en_input, de_input):
        labels = de_input.detach()
        en_pad_mask = pad_mask(en_input, self.device)
        de_combined_mask = combined_mask(de_input, self.device)
        de_pad_mask= pad_mask(de_input, self.device)

        en_input = self.embedding(en_input)
        de_input = self.embedding(de_input)

        hs = self.encoder(en_input, en_pad_mask)
        out = self.decoder(de_input, hs, de_combined_mask, de_pad_mask)
        out = self.fc(out)
        out = get_loss(out, labels)
        return out

    def search(self, en_input, de_input):
        en_pad_mask = pad_mask(en_input, self.device)
        de_combined_mask = combined_mask(de_input, self.device)
        de_pad_mask = pad_mask(de_input, self.device)

        en_input = self.embedding(en_input)
        de_input = self.embedding(de_input)

        hs = self.encoder(en_input, en_pad_mask)
        out = self.decoder(de_input, hs, de_combined_mask, de_pad_mask)
        out = self.fc(out)

        out = torch.argmax(out, dim=-1)
        return out[:, -1] # bs , 1


def get_loss(logits,labels):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    # <bos> token 제거
    labels = labels[:,1:].contiguous()
    logits = logits[:, :-1].contiguous()
    # loss 계산
    B, S = labels.size()
    losses = loss_fn(logits.view(B*S, -1), labels.view(-1))
    return losses

def pad_mask(input, device):
    bs, seq = input.size()
    zero_pad_mask = input.eq(0).float().to(device)
    return zero_pad_mask.view(bs,1, seq, 1)

def ahead_mask(seq, device):
    ahead = torch.ones(seq, seq).to(device)
    ahead = 1-torch.tril(ahead)
    return ahead

def combined_mask(input, device):
    bs, seq = input.size()
    ahead = ahead_mask(seq, device) # (seq, seq)
    pad = pad_mask(input, device) # bs, 1, 1, seq
    mask = ahead + pad
    return mask