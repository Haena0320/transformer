import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class embedding(nn.Module):
    def __init__(self, vocab_size, h_units, zero_pad=True, scale=True):
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.h_units = h_units
        self.zero_pad = zero_pad
        self.scale = scale
        self.lookup_table = torch.Tensor(self.vocab_size, self.h_units)

    def forward(self, inputs):
        if self.zero_pad:
            self.padding_idx = 0

        output = F.embedding(inputs, self.lookup_table, self.padding_idx)
        if self.scale:
            output = output*(self.h_units**0.5)

        return output


class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        super(layer_normalization, self).__init__()
        self.epsilon=epsilon
        self.gamma = torch.ones(features)
        self.betta = torch.zeros(features)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 각 input에 대해 normalization
        std = x.std(-1, keepdim=True)
        return self.gamma * (x-mean) / (std+self.epsilon)+self.betta

class positional_encoding(nn.Module):
    def __init__(self, h_units, zero_pad=True):

        super(positional_encoding, self).__init__()
        self.h_units = h_units
        self.zero_pad = zero_pad

    def forward(self, inputs):
        bs, seq = inputs.size() # batch size, sequence
        position_ind = torch.unsqueeze(torch.arange(0,seq), 0).repeat(bs, 1).long() # (batch_size, seq)
        position_enc = torch.Tensor([[ pos/ np.power(10000, 2.*i/self.h_units) for i in range(self.h_units)] for pos in range(seq)]) # (seq, h_units)

        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])

        padding_idx = 0
        outputs = F.embedding(position_ind, position_enc, padding_idx)

        return outputs


def sublayer_connection(input, sublayer_output, dropout_p):
    outputs = input + nn.Dropout(dropout_p)(sublayer_output)
    return outputs

def scaled_dot_product(query, key, value, key_masks, masked=False):
    # query * key
    outputs = torch.bmm(query, key.permute(0, 2, 1))  # (h*batch_size, sequence_q, sequence_k)
    # scaled
    outputs = outputs / (key.size()[-1] ** 0.5)

    padding = torch.ones(outputs.size()) * (-2 ** 32 + 1)
    condition = key_masks.eq(0.).float()
    outputs = padding * condition + outputs * (1. - condition)

    if masked:  # future blinding
        diag_vals = torch.ones(outputs[0, :, :].size())  # (sequence_q, sequence_k)
        tril = torch.tril(diag_vals, diagonal=0)  # (sequence_q, sequence_k)
        masks = torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1)

        padding = torch.ones(masks.size()) * (-2 ** 32 + 1)
        condition = masks.eq(0.).float()  # masking 될 곳 : 1
        outputs = padding * condition + outputs * (1. - condition)  # masking 될 곳 -2*32, 나머지값은 그대로 유지
        # 마스킹되는 곳은 소프트맥스 취하면 0으로 감.

    # activation
    outputs = F.softmax(outputs, dim=-1)  # (h*batch_size, sequence_q, sequence_k)

    # weighted sum
    outputs = torch.bmm(outputs, value)  # (h*bs, sequence_q, dimension/h)
    return outputs

class multihead_attention(nn.Module):
    def __init__(self, h_units, n_head=8,d_rate=0.1, masked=False):
        super(multihead_attention, self).__init__()
        self.h_units = h_units
        self.n_head = n_head
        self.d_rate = d_rate
        self.masked = masked

        self.Q_proj = nn.Sequential(nn.Linear(self.h_units, self.h_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.h_units, self.h_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.h_units, self.h_units), nn.ReLU())

        self.output_linear = nn.Sequential(nn.Linear(self.h_units,self.h_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p= self.d_rate)
        self.normalization = layer_normalization(self.h_units)

    def forward(self, queries, keys, values):
        # key, value, (batch_size, sequence_q(v),dimension)
        # queries (batch_size, sequence_k, dimension)

        Q = self.Q_proj(queries)
        K = self.K_proj(keys)
        V = self.V_proj(values)
        #make multi head
        Q_ = torch.cat(torch.chunk(Q, self.n_head, dim=2), dim=0) #(h*batch_size, sequence_q, dimension/h)
        K_ = torch.cat(torch.chunk(K, self.n_head, dim=2), dim=0) #(h*batch_size, sequence_k, dimension/h_
        V_ = torch.cat(torch.chunk(V, self.n_head, dim=2), dim=0) # (h*batch_size, sequence_v, dimension/h)

        key_masks = torch.sign(torch.abs(torch.sum(K, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.n_head, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)
        # (h*N, sequence_q, sequence_k)

        outputs = scaled_dot_product(Q_, K_, V_, key_masks, masked=self.masked)
        outputs = torch.cat(torch.chunk(outputs, self.n_head, dim=0), dim=2)  # (bs, sequence_q, dimension)
        outputs = self.output_linear(outputs)

        outputs = sublayer_connection(queries, outputs, self.d_rate)
        outputs = self.normalization.forward(outputs)

        return outputs


class feedforward(nn.Module):
    def __init__(self, in_channels, num_units=[2048, 512]):
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        self.fc1 = nn.Sequential(nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.num_units[0], self.num_units[1]))
        self.normalization = layer_normalization(self.in_channels)


    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.fc2(outputs)
        outputs = sublayer_connection(inputs, outputs,dropout_p=0.1)
        outputs = self.normalization.forward(outputs)

        return outputs

class label_smoothing(nn.Module):

    def __init__(self, epsilon=0.1):
        super(label_smoothing, self).__init__()
        self.epsilon=epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return ((1-self.epsilon)*inputs)+(self.epsilon/K)


if __name__=="__main__":
    num_units=512
    inputs = torch.randn((100, 10))  # batch_size, sequence
    outputs = positional_encoding(num_units)(inputs)
    outputs = multihead_attention(num_units)(outputs, outputs, outputs)
    outputs = feedforward(num_units)(outputs)
    print(outputs)
























