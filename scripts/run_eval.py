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
parser.add_argument("--use_pretrained", type=int, default=0)
parser.add_argument("--total_steps", type=int, default=10000)
parser.add_argument("--eval_period", type=int, default=1000)

args = parser.parse_args()
assert args.model in ['base', "large"]
assert args.dataset in ["en_de", "en_fr"]

config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")
print("current device {}".format(device))
## model load -> valid -> best_ckpnt
ckpnt_loc = "./log/0.001/ckpnt/"+"ckpnt_{}".format(args.use_pretrained)
checkpoint = torch.load(ckpnt_loc, map_location=device)


# model load
vocab = config.data_info[args.dataset].vocab_size
d_model = config.model.h_units
num_heads = config.model.n_head
num_layers = config.model.n_blocks
d_ff = config.model.dim_feedforward
dropout = config.model.d_rate

#model = TransformerModel(vocab, d_model, dropout, num_layers, d_model, d_ff, num_heads, device)
model  = TransformerModel(config, args, device)
model.to(device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
## data load -> test data load
data = config.data_info[args.dataset]
data_list = [data.prepro_te_en, data.prepro_te_de]
data_loader = get_data_loader(data_list, 10, shuffle=False)

# bleu score calculation
sp = spm.SentencePieceProcessor()
sp.Load("bye_pair_encoding.model")

import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang="du")
total_bleu = 0
cnt = 0
for data_iter in tqdm(data_loader):
    encoder_input = data_iter["encoder"].to(device) #(16, 100)
    decoder_input = data_iter["decoder"].to(device) #(16, 100)

    sos = decoder_input[:,0] # bs
    sos = sos.unsqueeze(1) # (bs, 1)

    bs, max_sent_len = decoder_input.size()
    max_sent_len += 50

    pred_token = torch.zeros(bs, max_sent_len)
    for i in range(max_sent_len):
        y = model.search(encoder_input, sos) # bs,
        pred_token[:, i] = y
        sos = torch.cat([sos, y.unsqueeze(1)], dim=-1)

    pred_token = pred_token.tolist()
    for i, token in enumerate(pred_token):
        for j in range(len(token)):
            if token[j] == 2:
                token = token[:j]
                break
        print("-----------------------------------------------------")
        token = [int(t) for t in token]
        decode_tokens = sp.DecodeIds(token)
        decode_truth = sp.DecodeIds(decoder_input[i,:].tolist())
        print(decode_tokens)
        print(decode_truth)
        #pred = md.detokenize(decode_tokens.strip().split())
        #truth = md.detokenize(decode_truth.strip().split())
        bleu = sacrebleu.corpus_bleu(decode_tokens, decode_truth, force=True, lowercase=True)
        total_bleu+=bleu.score
        cnt+= 1
        print("bleu {}".format(bleu.score))

print('-----------------------------------------------------------------------------------------------------------------')
print("total bleu :{}".format(total_bleu/cnt))




# 1(sos) + token_id, token_id, token_id, + 2(eos)
# sp = torch.load("./data/prepro/en_de/word2id_de.pkl")
# sp = {v:k for k, v in sp.items()}
# import sacrebleu
# from sacremoses import MosesDetokenizer
# md = MosesDetokenizer(lang="du")
# total_bleu = 0
# bleu_cnt = 0
# for data_iter in tqdm(data_loader):
#     encoder_input = data_iter["encoder"].to(device) #(16, 100)
#     decoder_input = data_iter["decoder"].to(device) #(16, 100)
#
#
#     sos = decoder_input[:,0] # bs
#     sos = sos.unsqueeze(1) # (bs, 1)
#
#     bs, max_sent_len = decoder_input.size()
#     max_sent_len += 50
#
#     pred_token = torch.zeros(bs, max_sent_len)
#     for i in range(max_sent_len):
#         y = model.search(encoder_input, sos) # bs,
#         pred_token[:, i] = y
#         sos = torch.cat([sos, y.unsqueeze(1)], dim=-1)
#
#     pred_token = pred_token.tolist()
#
#     for i, token in enumerate(pred_token):
#         pred_ = list()
#         for j in range(len(token)):
#             if token[j] == 3:
#                 break
#             elif token[j] == 1:
#                 continue
#             else:
#                 pred_.append(token[j])
#         print("-----------------------------------------------------")
#         pred_ = [int(t) for t in pred_]
#         pred_ = [sp[t] for t in pred_]
#         pred_ = " ".join(pred_)
#         print(pred_)
#
#         decode_truth = decoder_input[i, 1:].tolist()
#         decode_truth = [int(d) for d in decode_truth]
#         truth_ = list()
#         for d in range(len(decode_truth)):
#             if decode_truth[d] == 3:
#                 break
#             elif decode_truth[d] == 1:
#                 continue
#             else:
#                 truth_.append(decode_truth[d])
#
#         truth_ = [sp[t] for t in truth_]
#         truth_ = " ".join(truth_)
#         print(truth_)
#
#         pred = md.detokenize(pred_.strip().split())
#         truth = md.detokenize(truth_.strip().split())
#         bleu = sacrebleu.corpus_bleu(pred, truth)
#         total_bleu += bleu.score
#         bleu_cnt += 1
#
#         print("bleu {}".format(bleu.score))
#        # print("total bleu {}".format(total_bleu))
#
# print('-----------------------------------------------------------------------------------------------------------------')
#
# print("total bleu :{}".format(total_bleu/bleu_cnt))