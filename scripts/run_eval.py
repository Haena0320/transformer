import nltk.translate.bleu_score as bleu
import sys, os
sys.path.append(os.getcwd())
import torch
import argparse
from src.utils import *
from src.data_load import *
from tqdm import tqdm
import sentencepiece as spm
from src.model_2 import *

parser = argparse.ArgumentParser()
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
print("current device {}".format(device))
## model load -> valid -> best_ckpnt
checkpoint = torch.load("./log/0.001/ckpnt/ckpnt_16", map_location=device)
model  = TransformerModel(config, args, device)
model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
## data load -> test data load
data = config.data_info[args.dataset]
data_list = [data.prepro_te_en, data.prepro_te_de]
data_loader = get_data_loader(data_list, config.train.batch_size)

# 1(sos) + token_id, token_id, token_id, + 2(eos)
## decoding

## bleu score calculation
sp = spm.SentencePieceProcessor()
sp.Load("bye_pair_encoding.model")

pred = open("./eval/predict.txt", "w")
truth = open("./eval/truth.txt", "w")

for data_iter in tqdm(data_loader):
    encoder_input = data_iter["encoder"].to(device)
    decoder_input = data_iter["decoder"].to(device)

    sos = decoder_input[:,0]
    sos = sos.unsqueeze(1)

    bs, max_sent_len = decoder_input.size()
    max_sent_len += 50

    generate_data = []
    for i in range(max_sent_len):
        y = model.search(encoder_input, sos)
        sos = torch.cat([sos, y], dim=-1)

    tokens = sos.tolist()
    for i, token in enumerate(tokens):
        for j in range(len(token)):
            if token[j] == 2:
                token = token[:j]
                break

        decode_tokens = sp.DecodeIds(token)
        decode_truth = sp.DecodeIds(decoder_input[i,:].tolist())
        # write txt file
        pred.write(decode_tokens+"\n")
        truth.write(decode_truth+"\n")

pred.close()
truth.close()
print("prediction finished..")
############################################ blue score calculation ####################################################
import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang="du")


p = open("./eval/predict.txt", "r")
t = open("./eval/truth.txt", "r")

pred = [i.replace("\n", '') for i in p.readlines()]
pred = [md.detokenize(i.strip().split()) for i in pred]
truth = [i.replace("\n", '') for i in t.readlines()]
truth = [md.detokenize(i.strip().split()) for i in truth]

print("sample 1st pred sentence:", pred[0])
print("sample 1st truth sentence:", truth[0])

assert len(pred) == len(truth)
print(len(pred))
print(len(truth))

total_bleu = []
for i in range(len(pred)):
    bleu = sacrebleu.corpus_bleu(pred[i],truth[i])
    total_bleu.append(bleu.score)

print('-----------------------------------------------------------------------------------------------------------------')
print("total bleu :{}".format(sum(total_bleu)/len(total_bleu)))