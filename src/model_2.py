import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, LayerNorm
import math

class WordEncoding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(WordEncoding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def init_weights(self):
        d = self.d_model**(-0.5)
        self.embedding.weight.data.uniform_(-d, d)

    def forward(self,x):
        return self.embedding(x)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=None):
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

class ScaledDotProduct(nn.Module):
    def __init__(self, dropout= 0.1, device=None):
        super(ScaledDotProduct, self).__init__()
        self.dropout= dropout
        self.device = device

    def forward(self, query, key, value, attn_mask=None):
        # (bs * h, seq_q, dimension/h) (128, 56, 64)
        # (bs * h, seq_k, dimension/h)
        # (bs * h, seq_v, dimension/h)
        bs_h, K, dimension_k = key.size()
        _, Q, _ = query.size()
        attn_output = torch.bmm(query, key.transpose(1,2))/dimension_k**0.5 # attn_output : (bs, seq_q, seq_k)
        if attn_mask is not None:
            mask = torch.ones(Q,K)
            mask = 1-torch.tril(mask, diagonal=0)
            mask = mask*(-2**32)
            mask = mask.repeat(bs_h, 1, 1)
            device = torch.device("cuda:0")
            attn_output *=  mask.to(device)
            #attn_output += mask
        attn_output = F.softmax(attn_output, dim=-1)
        print(attn_output[0,:,:])
        attn_output = F.dropout(attn_output, p=self.dropout)
        output = torch.bmm(attn_output,value) # output : (bs, seq_q, d_model)

        return output


class MultiheadAttention_In(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention_In, self).__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

    def init_weight(self):
        self.fc_q = self.fc_q.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_k = self.fc_k.weight.data.normal_(mean=0.0, std=0.02)
        self.fc_v = self.fc_v.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, query, key, value): # (bs, seq, embedding_dim)
        bs, seq, d_model = query.size()
        head_dim = d_model// self.num_heads
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
        self.linear = nn.Linear(d_model, d_model)

    def init_weights(self):
        self.linear.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, attn_output):
        bs_h,seq,_ = attn_output.size()
        bs = bs_h // self.num_heads
        assert bs * self.num_heads == bs_h

        attn_output = torch.cat(torch.chunk(attn_output,self.num_heads, dim=0), dim=2) # (bs,seq_q,embedding_dim)
        return self.linear(attn_output)

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
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, input, input_mask=None):
        query, key, value = self.attn_in(input, input, input)
        attn_out = self.scaled_dot(query, key, value, input_mask)
        out1 = self.attn_out(attn_out)
        out = self.norm1(input+self.dropout1(out1))
        out2= self.linear2(self.dropout(self.activation(self.linear1(out))))
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
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm3.bias.data.zero_()
        self.norm3.weight.data.fill_(1.0)

    def forward(self, input, enc, input_mask=None):
        query, key, value = self.attn_in_1(input, input, input)
        attn_out1 = self.scaled_dot_1(query, key, value, input_mask) # input_mask : (bs, seq_q, seq_k)
        out1 = self.attn_out_1(attn_out1)
        out = self.norm1(input + self.dropout1(out1))

        query, key, value = self.attn_in_2(out,enc,enc)
        attn_out2 = self.scaled_dot_2(query, key, value, None)
        out2 = self.attn_out_2(attn_out2)
        out = self.norm2(out+self.dropout2(out2))

        out3= self.linear2(self.dropout(self.activation(self.linear1(out))))
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
        for layer in self.layers:
            output = layer(input, input_mask=mask)

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
        for layer in self.layers:
            output = layer(input, enc, input_mask=mask)

        return output

class Embedding(nn.Module):
    def __init__(self,d_model, vocab, device):
        super(Embedding, self).__init__()
        self.word_embed = WordEncoding(d_model, vocab).to(device)
        self.posit_embed = PositionEncoding(d_model=d_model, device=device)
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(p=0.1)

    def init_weights(self):
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.word_embed.init_weights()

    def forward(self, input):
        output = self.word_embed(input)+self.posit_embed(input)
        return self.dropout(self.norm(output))


class TransformerModel(nn.Module):
    def __init__(self, config, args, device):
        vocab = config.data_info[args.dataset].vocab_size
        d_model = config.model.h_units
        num_heads = config.model.n_head
        num_layers = config.model.n_blocks
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.d_rate

        super(TransformerModel, self).__init__()
        self.emb = Embedding(d_model, vocab, device)
        encoder = EncoderLayer(d_model, num_heads, dim_feedforward, dropout, device)
        self.enc = TransformerEncoder(encoder, num_layers, device)

        decoder = DecoderLayer(d_model, num_heads, dim_feedforward, dropout, device)
        self.dec = TransformerDecoder(decoder, num_layers, device)
        self.criterion = nn.Linear(d_model, vocab)

    def init_weights(self):
        self.emb.init_weights()
        self.enc.init_weights()
        self.dec.init_weights()


    def forward(self, x, y):
        # <eos> token 제거
        dec = y * (1 - y.eq(2.).float())
        dec = dec[:, :-1].long()

        # forward
        enc_emb = self.emb(x)
        enc_output = self.enc(enc_emb)
        dec_emb = self.emb(dec)
        dec_output = self.dec(dec_emb, enc_output, mask=1)
        dec_output = self.criterion(dec_output)
        loss = get_loss(y, dec_output)
        return loss

    def search(self, x, y):

        # encoding forward
        enc_emb = self.emb(x)
        enc_output = self.enc(enc_emb)

        # auto regressive
        dec_emb = self.emb(y)
        dec_output = self.dec(dec_emb, enc_output, mask=1)
        dec_output = self.criterion(dec_output) # (bs, seq_len, vocab_size)

        # decoding
        output = torch.argmax(dec_output, dim=-1)
        return output[:, -1].unsqueeze(1)

def get_loss(labels, logits):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    # <bos> token 제거
    label = labels[:,1:].contiguous()
    # loss 계산
    B, S = label.size()
    losses = loss_fn(logits.view(B*S, -1), label.view(-1))
    return losses

















