from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
import sys, os
import torch
import argparse
from src.utils import *

parse = argparse.ArgumentParser()
parser.add_argument("--mode",type=str, default="><")
parser.add_argument("--dataset", type=str, default="en_de")
parser.add_argument("--model", type=str, default="base")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--total_steps", type=int, default=10000)
parser.add_argument("--eval_period", type=int, default=1000)

parser.add_argument("--load", type=str, default="model.1.ckpt")

args = parser.parse_args()
assert args.model in ['base', "large"]
assert args.dataset in ["en_de", "en_fr"]

config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

## model load -> valid -> best_ckpnt
torch.load()
## data load -> test data load

## make batch

## decoding

## evaluation(bleu)


