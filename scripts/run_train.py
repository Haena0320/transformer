import sys, os
sys.path.append(os.getcwd())
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from src.utils import *
from src.train import *
from src.prepro import *

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default=">____<")
parser.add_argument("--log", type=str, default="loss")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--dataset", type=str, default="en_de")
parser.add_argument("--model", type=str, default="base")

parser.add_argument("--att_heads", type=int, default=4)
parser.add_argument("--m_h_units", type=int, default=128)
parser.add_argument("--f_h_units", type=int, default=512)
parser.add_argument('--layer_size', type=int, default=2)

parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--h_units", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--g_norm", type=str, default=5)
parser.add_argument("--learning_rate", type=float,default=1e-3)

parser.add_argument("--use_earlystop", type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)

args = parser.parse_args()
config = load_config(args.config)

assert args.model in ["base", "large"]
assert args.dataset in ["en_de", "en_fr"]

# gpu / cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else 'cpu')

# log save file
oj = os.path.join

args.log = "./log/{}".format(args.learning_rate)
tb_loc = oj(args.log, 'tb')
ckpnt_loc = oj(args.log, "ckpnt")

if not os.path.exists(args.log):
    os.mkdir(args.log)
    os.mkdir(tb_loc)
    os.mkdir(ckpnt_loc)
    
writer = SummaryWriter(tb_loc)

from src.model import Transformer as model
import src.train as train

# vocab
de2idx = torch.load(config.prepro_vocab.de)
en2idx = torch.load(config.prepro_vocab.en)
enc_voc = len(de2idx)
dec_voc = len(en2idx)

# data loader
train_data = torch.load(config.path_prepro.en_de[0])
test_data = torch.load(config.path_prepro.en_de[1])

train_loader = get_batch_indices(len(train_data["data"]), config.train.batch_size, train_data)
test_loader = get_batch_indices(len(test_data["data"]), config.train.batch_size, test_data)
#print("dataset iteration num : train {} | test {}".format(len(train_loader), len(test_loader)))

# model load
model = model(config, args)

# trainer load
trainer = train.get_trainer(config, args,device, train_loader, writer, "train")
#dev_trainer = train.get_trainer(config, args,device, dev_loader, writer, "dev")
test_loader = train.get_trainer(config, args,device, test_loader, writer, "test")

optimizer = train.get_optimizer(model, args.optim)
schedular = train.get_lr_schedular(optimizer)

trainer.init_optimizer(optimizer)
trainer.init_schedular(schedular)

early_stop_loss = []
total_epoch = args.epochs * 100
for epoch in tqdm(range(1, args.epochs+1)):
    trainer.train_epoch(model, epoch)
    valid_loss = dev_trainer.train_epoch(model, epoch, trainer.global_step)
    early_stop_loss.extend(valid_loss)

    if args.use_earlystop and early_stop_loss[-2] < early_stop_loss[-1]:
        break
    ### torch model, param, save

    train_eval = trainer.evaluator
    valid_eval = dev_trainer.evaluator
    print("epoch : {} | train_eval : {} | valid_eval : {}".format(epoch, train_eval, valid_eval))


print('train finished...')
# test evaluation

print("finished !! ")

    



