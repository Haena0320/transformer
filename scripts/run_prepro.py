import os, sys
sys.path.append(os.getcwd())
import torch
import argparse
from src.utils import *
from src.prepro import *
from src.data_load import *
import logging
import sentencepiece as spm
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="Q_Q")
args.add_argument("--dataset", type=str, default="en_de")
args.add_argument("--model", type=str, default="base")
args.add_argument("--config", type=str, default="default")

args = args.parse_args()
config = load_config(args.config)

assert args.model in ["base", "large"]
assert args.dataset in ["en_de", "en_fr"]

logging.info("Building vocab ")
data_info = config.data_info[args.dataset]
encoding(data_info.raw_tr_total, data_info.vocab_size, data_info.vocab_path,  data_info.model, data_info.model_type)

logging.info("make data ! ")
raw =[data_info.raw_tr_de, data_info.raw_tr_en, data_info.raw_te_de, data_info.raw_te_en]
prepro = [data_info.prepro_tr_de, data_info.prepro_tr_en, data_info.prepro_te_de, data_info.prepro_te_en]

for input, output in tqdm(list(zip(raw, prepro))):
    data_prepro(input, output, data_info.model+".model")

print('finished !! ')

