import torch
import torch.nn as nn

alpha = 1.2
betha = 5

class BeamSearch():
    def __init__(self, device, beam_size, max_length):
        self.beam_size = beam_size
        self.device = device
        self.max_length = max_length
        super(BeamSearch, self).__init__()
        # 초기 time step 별 인덱스 초기화
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device)+data_loaer.BOS]
        self.prev_beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device)-1]
        self.cumulative_probs = [torch.LongTensor([.0]+[-float('inf')]*(beam_size-1)).to(self.device)]
        self.masks = [torch.ByteTensor(beam_size).zero_().to(self.device)]



    def is_done(self):
        if self.done_cnt >= self.beam_size:
            return 1
        else:
            return 0

def length_penalty(length, alpha, betha):
    l = (length+betha) / (betha + 1)
    return l** alpha

def reverse_dict(dict):
    reverse_dict = {v:k for k, v in dict.items()}
    return reverse_dict


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return


