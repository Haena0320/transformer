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
        self.d_model = d_model

    def forward(self, inputs):
        bs, seq = inputs.size()
        # batch size, sequence
        self.position_emb = torch.zeros(seq, self.d_model, requires_grad=False)
        position = torch.arange(0., seq).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * - (math.log(1e4) / self.d_model))
        self.position_emb[:, 0::2] = torch.sin(position * div_term)
        self.position_emb[:, 1::2] = torch.cos(position * div_term)
        outputs = self.position_emb.unsqueeze(0).repeat(inputs.size(0), 1, 1).to(self.device)
        return outputs

class ScaledDotProduct(nn.Module):
    def __init__(self, dropout= 0.1, device=None):
        super(ScaledDotProduct, self).__init__()
        self.dropout= dropout
        self.device = device

    def forward(self, query, key, value, attn_mask=None, decod_mask=None):
        # (bs * h, seq_q, dimension/h) (128, 56, 64)
        # (bs * h, seq_k, dimension/h)
        # (bs * h, seq_v, dimension/h)
        # attn_mask (bs, seq_k)
        bs_h, K, dimension_k = key.size()
        _, Q, _ = query.size()
        attn_output = torch.bmm(query, key.transpose(1,2))/dimension_k**0.5 # attn_output : (bs*h, seq_q, seq_k)
        # padd mask
        bs, _ = attn_mask.size()
        h = bs_h // bs
        assert bs_h == h * bs
        attn_mask = attn_mask.eq(0).float().unsqueeze(1).repeat(1,Q,1).repeat(h,1,1).contiguous() #(bs*h,Q,seq_k)
        attn_output += attn_mask*(-1e10)

        if decod_mask is not None: # decoder mask
            mask = torch.ones(Q,K, device="cuda:0",requires_grad=False)
            mask = 1-torch.tril(mask, diagonal=0)
            mask = mask.unsqueeze(0).repeat(bs_h,1,1).contiguous()
            a_mask = mask * (-1e10)
            attn_output = attn_output + a_mask.to(torch.device("cuda:0"))

        attn_output = F.softmax(attn_output, dim=-1)
        attn_output = F.dropout(attn_output, p=self.dropout)
        output = torch.bmm(attn_output,value) # output : (bs, seq_q, d_model)

        return output


class MultiheadAttention_In(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention_In, self).__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(d_model, d_model, bias=False)
        self.fc_k = nn.Linear(d_model, d_model, bias=False)
        self.fc_v = nn.Linear(d_model, d_model, bias=False)

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc_q.weight.data)
        nn.init.xavier_uniform_(self.fc_k.weight.data)
        nn.init.xavier_uniform_(self.fc_v.weight.data)

    def forward(self, query, key, value): # (bs, seq, embedding_dim)
        bs, seq, d_model = query.size()
        head_dim = d_model // self.num_heads
        assert head_dim * self.num_heads == d_model
        q = self.fc_q(query)
        q = torch.cat(torch.chunk(q, self.num_heads, dim=2), dim=0).contiguous()

        k = self.fc_k(key)
        k = torch.cat(torch.chunk(k, self.num_heads, dim=2), dim=0).contiguous()

        v = self.fc_v(value)
        v = torch.cat(torch.chunk(v, self.num_heads, dim=2), dim=0).contiguous()
        return q, k, v

class MultiheadAttention_Out(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention_Out, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight.data)

    def forward(self, attn_output):
        bs_h,seq,_ = attn_output.size()
        bs = bs_h // self.num_heads
        assert bs * self.num_heads == bs_h

        attn_output = torch.cat(torch.chunk(attn_output,self.num_heads, dim=0), dim=2).contiguous() # (bs,seq_q,embedding_dim)
        attn_output = self.linear(attn_output)
        return attn_output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu", device=None):
        super(EncoderLayer, self).__init__()
        self.attn_in = MultiheadAttention_In(d_model, num_heads)
        self.scaled_dot = ScaledDotProduct(dropout=dropout, device=device)
        self.attn_out = MultiheadAttention_Out(d_model, num_heads)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = nn.ReLU()

    def init_weights(self):
        self.attn_in.init_weights()
        self.attn_out.init_weights()
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, input, mask=None):
        query, key, value = self.attn_in(input, input, input)
        attn_out = self.scaled_dot(query, key, value, attn_mask=mask)
        out1 = self.attn_out(attn_out)
        out = self.norm1(input+self.dropout1(out1))
        out2= self.linear2(self.activation(self.linear1(out)))
        out = self.norm2(out1+self.dropout2(out2))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu", device=None):
        super(DecoderLayer, self).__init__()
        self.attn_in_1 = MultiheadAttention_In(d_model, num_heads)
        self.scaled_dot_1 = ScaledDotProduct(dropout=dropout, device=device)
        self.attn_out_1 = MultiheadAttention_Out(d_model, num_heads)

        self.attn_in_2 = MultiheadAttention_In(d_model, num_heads)
        self.scaled_dot_2 = ScaledDotProduct(dropout=dropout, device=device)
        self.attn_out_2 = MultiheadAttention_Out(d_model, num_heads)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)


        self.activation = nn.ReLU()
        self.device = device

    def init_weights(self):
        self.attn_in_1.init_weights()
        self.attn_out_1.init_weights()
        self.attn_in_2.init_weights()
        self.attn_out_2.init_weights()

        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm3.bias.data.zero_()
        self.norm3.weight.data.fill_(1.0)

    def forward(self, input, enc, mask=None):
        query, key, value = self.attn_in_1(input, input, input)
        attn_out1 = self.scaled_dot_1(query, key, value, attn_mask=mask[0], decod_mask=1) # input_mask : (bs, seq_q, seq_k)
        out1 = self.attn_out_1(attn_out1)
        out = self.norm1(input + self.dropout1(out1))

        query, key, value = self.attn_in_2(out,enc,enc)
        attn_out2 = self.scaled_dot_2(query, key, value, attn_mask=mask[1])
        out2 = self.attn_out_2(attn_out2)
        out = self.norm2(out+self.dropout2(out2))

        out3= self.linear2(self.activation(self.linear1(out)))
        out = self.norm3(out+self.dropout3(out3))
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, device):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.layers.to(device)
        self.num_layers = num_layers

    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()

    def forward(self, input, mask=None):
        output = input
        for layer in self.layers:
            output = layer(output, mask=mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers, device):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.layers.to(device)
        self.num_layers = num_layers


    def init_weights(self):
        for layer in self.layers:
            layer.init_weights()

    def forward(self, input,enc, mask=None):
        output = input
        for layer in self.layers:
            output = layer(input, enc, mask=mask)
        return output

class Embedding(nn.Module):
    def __init__(self,d_model, embed_weights, device):
        super(Embedding, self).__init__()
        self.word_embed = WordEncoding(embed_weights, d_model).to(device)
        self.posit_embed = PositionEncoding(d_model=d_model, device=device)
        #self.norm = LayerNorm(d_model)
        self.dropout = Dropout(p=0.1)

    def forward(self, input):
        output = self.word_embed(input)+self.posit_embed(input)
        return self.dropout(output)


class TransformerModel(nn.Module):
    def __init__(self, config, args, device):
        vocab = config.data_info[args.dataset].vocab_size
        d_model = config.model.h_units
        num_heads = config.model.n_head
        num_layers = config.model.n_blocks
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.d_rate

        super(TransformerModel, self).__init__()
        emb_weights = nn.parameter.Parameter(torch.empty(vocab,d_model), requires_grad=True)
        nn.init.normal_(emb_weights, mean=0, std=d_model**(-0.5))

        self.emb = Embedding(d_model, emb_weights, device)
        encoder = EncoderLayer(d_model, num_heads, dim_feedforward, dropout, device)
        self.enc = TransformerEncoder(encoder, num_layers, device)

        decoder = DecoderLayer(d_model, num_heads, dim_feedforward, dropout, device)
        self.dec = TransformerDecoder(decoder, num_layers, device)
        self.criterion = nn.Linear(d_model, vocab, bias=False)

    def init_weights(self):
        self.enc.init_weights()
        self.dec.init_weights()
        self.criterion.weight_ = self.emb.T

    def forward(self, x, y):
        """
        :param x: (bs, max_length-1)
        :param y: (bs, max_length)
        :return: (bs, max_length-1)
        """
        # forward
        enc_emb = self.emb(x)
        enc_output = self.enc(enc_emb, mask=x)
        dec_emb = self.emb(y)
        dec_output = self.dec(dec_emb, enc_output, mask=[y, x])
        dec_output = self.criterion(dec_output)
        loss = get_loss(y, dec_output)
        return loss

    def search(self, x, y):
        """
        :param x: (bs, max_length-1)
        :param y: (bs, n)
        :return: (bs, n)
        """
        # encoding forward
        enc_emb = self.emb(x)
        enc_output = self.enc(enc_emb, mask=x)

        # auto regressive
        dec_emb = self.emb(y)
        dec_output = self.dec(dec_emb, enc_output, mask=[y, x])
        dec_output = self.criterion(dec_output) # (bs, n, vocab_size)

        # decoding
        output = torch.argmax(dec_output, dim=-1) # bs, n
        return output[:, -1]# bs, 1

def get_loss(labels, logits):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    # <bos> token 제거
    labels = labels[:,1:].contiguous()
    logits = logits[:, :-1].contiguous()
    # loss 계산
    B, S = labels.size()
    losses = loss_fn(logits.view(B*S, -1), labels.view(-1))
    return losses
