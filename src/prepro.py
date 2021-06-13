import logging
import numpy as np
import sentencepiece as spm
import torch

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

def data_prepro(input_path, save_path, model_path):
    f1 = open(input_path[0])
    f2 = open(input_path[1])
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    sp.SetEncodeExtraOptions("bos:eos")
    
    en_line = [line.split("\n")[0] for line in f1.readlines()]
    de_line = [line.split("\n")[0] for line in f2.readlines()]

    print(en_line[-10:])
    print(de_line[-10:])

    ids1 = [np.array(sp.EncodeAsIds(line)) for line in en_line]
    ids2 = [np.array(sp.EncodeAsIds(line)) for line in de_line]


    print(ids1[:10])
    print(ids2[:10])
    print("english prepro dataset : {}".format(len(ids1)))
    print("german prepro dataset : {}".format(len(ids2)))

    torch.save(ids1, save_path[0])
    torch.save(ids2, save_path[1])
    logging.info("data saved ! ")
    
