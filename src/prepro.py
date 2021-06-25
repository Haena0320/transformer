from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import logging
import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm


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


def encoding(input_file, vocab_size, vocab_path, model_name, model_type):
    pad = 0
    bos = 1
    eos = 2
    unk = 3
    character_coverage = 1.0
    input_argument = "--input=%s --model_prefix=%s --vocab_size=%s --character_coverage=%s --pad_id=%s --bos_id=%s --eos_id=%s --unk_id=%s --model_type=%s"
    cmd = input_argument % (input_file, model_name, vocab_size,character_coverage, pad, bos, eos, unk, model_type)

    spm.SentencePieceTrainer.Train(cmd)
    logging.info("model, vocab finished ! ")


def data_prepro(input_path, save_path, model_path):
    f_en = open(input_path[0], "r")
    f_de = open(input_path[1], "r")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    sp.SetEncodeExtraOptions("bos:eos")
    words1 = [line.split("\n")[0] for line in f_en.readlines()]
    words2 = [line.split("\n")[0] for line in f_de.readlines()]
    print(words1[0])
    print(words2[0])

    en_prepro = list()
    de_prepro = list()
    for en, de in tqdm(list(zip(words1, words2)), desc="make data"):
        en_ = np.array(sp.EncodeAsIds(en))
        de_ = np.array(sp.EncodeAsIds(de))

        if len(en_) > 1.7*len(en_) or len(de_)*1.7 < len(en_):
            continue
        else:
            en_prepro.append(en_)
            de_prepro.append(de_)

    assert len(en_prepro) == len(de_prepro)
    torch.save(en_prepro, save_path[0])
    torch.save(de_prepro, save_path[1])
    logging.info("data saved ! ")


def data_load(file):
    with open(file, "r", encoding="utf8") as f:
        data = f.readlines()
        data = [d.split("\n")[0] for d in data]
    return data




def make_word2id(file): # pad : 0, unk :1, start:2, end : 3
    vocab = data_load(file)
    word2id = {"<pad>":0}
    for word in vocab:
        word2id[word] = len(word2id)
    return word2id


def make_ids(file_path, word2id):
    prepro = list()
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split("\n")[0]
        words =["<s>"] +line.split(" ") + ["</s>"]
        print(words)
        ids = [word2id.get(word,1) for word in words]
        prepro.append(ids)
    return prepro


# word2id_en = make_word2id("/data/user15/workspace/Transformer/data/raw/en_de/vocab/vocab.50K.en.txt")
# word2id_de = make_word2id("/data/user15/workspace/Transformer/data/raw/en_de/vocab/vocab.50K.de.txt")
#
# train_en = make_ids("/data/user15/workspace/Transformer/data/raw/en_de/train/train.en.txt", word2id_en)
# train_de = make_ids("/data/user15/workspace/Transformer/data/raw/en_de/train/train.de.txt", word2id_de)
#
# test_en = make_ids('/data/user15/workspace/Transformer/data/raw/en_de/test/newstest2014.en.txt', word2id_en)
# test_de = make_ids("/data/user15/workspace/Transformer/data/raw/en_de/test/newstest2014.de.txt", word2id_de)
#
# import torch
# torch.save(word2id_en, "/data/user15/workspace/Transformer/data/prepro/en_de/word2id_en.pkl")
# torch.save(word2id_de, "/data/user15/workspace/Transformer/data/prepro/en_de/word2id_de.pkl")
# torch.save(train_en, "/data/user15/workspace/Transformer/data/prepro/en_de/train.en.txt")
# torch.save(train_de, "/data/user15/workspace/Transformer/data/prepro/en_de/train.de.txt")
# torch.save(test_en, "/data/user15/workspace/Transformer/data/prepro/en_de/test.en.txt")
# torch.save(test_de, "/data/user15/workspace/Transformer/data/prepro/en_de/test.de.txt")
#
#