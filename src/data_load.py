from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import logging
import numpy as np

class Dataloader(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __getitem__(self, idx):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab


    
    
    
import sentencepiece as spm

input_file = '/hdd/user15/workspace/Transformer/data/raw/en_de/train/train.sum.txt'
vocab_size = 37000
model_name = "bye_pair_encoding"
model_type='bpe'
user_defined_symbols = "[PAD],[UNK],[SEP],[MASK]"
input_argument = "--input=%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s"
cmd = input_argument%(input_file, model_name, vocab_size, user_defined_symbols, model_type)
spm.SentencePieceTrainer.Train(cmd)

sp = spm.SentencePieceProcessor()
sp.Load("bye_pair_encoding.model")
f= open("/hdd/user15/workspace/Transformer/data/raw/en_de/train/train.de.txt")
ids = [np.array(sp.EncodeAsIds(line)) for line in f.readlines()]
import torch
torch.save(ids,"./data/prepro/en_de/train.pkl")

input_file = './~~ '# en_fr
vocab_size = 32000
model_name = "word_piece_encoding"
model_type="word"


def encoding(input_file, vocab_size, vocab_path, model_name, model_type):
    pad = 0
    bos = 1
    eos = 2
    unk = 3
    input_argument = "--input=%s --model_prefix=%s --vocab_size=%s --pad_id=%s --bos_id=%s --eos_id=%s --unk_id=%s --model_type=%s"
    cmd = input_argument % (input_file, model_name, vocab_size, pad, bos, eos, unk, model_type)

    spm.SentencePieceTrainer.Train(cmd)
    logging.info("model, vocab finished ! ")
    f = open(model_name+".vocab", encoding="utf-8")
    v = [doc.strip().split("\t") for doc in f]
    word2idx = {w[0]: i for i, w in enumerate(v)}
    torch.save(word2idx, vocab_path)

def data_prepro(input_path, save_path, model_path):
    f = open(input_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    sp.SetEncodeExtraOptions("bos:eos")
    ids = [np.array(sp.EncodeAsIds(line)) for line in f.readlines()]
    torch.save(ids, save_path)
    logging.info("data saved ! ")









    