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
parser.add_argument("--gpu", type=str, default='0')
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

from src.model_2 import TransformerModel as model
import src.train as train
import src.data_load as data

# data loader
data_info = config.data_info[args.dataset]
train_loader  = data.get_data_loader([data_info.prepro_tr_en, data_info.prepro_tr_de], config.train.batch_size ,False, 10, True)
#valid_loader = data.get_data_loader(data_list, config.train.batch_size, False, 10, True)
test_loader = data.get_data_loader([data_info.prepro_te_en, data_info.prepro_te_de], config.train.batch_size, False, 10, True)

print("dataset iteration num : train {} | test {}".format(len(train_loader), len(test_loader)))

# model load
model = model(config, args, device)

# trainer load
trainer = train.get_trainer(config, args,device, train_loader, writer, "train")
#dev_trainer = train.get_trainer(config, args,device, dev_loader, writer, "dev")
test_trainer = train.get_trainer(config, args,device, test_loader, writer, "test")

total_steps = args.epochs * len(train_loader)
optimizer = train.get_optimizer(model, args.optim)
schedular = train.get_lr_schedular(optimizer, config)

trainer.init_optimizer(optimizer)
trainer.init_schedular(schedular)

early_stop_loss = []
total_epoch = 100000//len(train_loader)
print("total epoch {}".format(total_epoch))
for epoch in tqdm(range(1, args.epochs+1)):
    trainer.train_epoch(model, epoch, save_path=ckpnt_loc+"/ckpnt.pkl")


    #test_loss = test_trainer.train_epoch(model, epoch, trainer.global_step)
    #early_stop_loss.extend(test_loss)

    # if args.use_earlystop and early_stop_loss[-2] < early_stop_loss[-1]:
    #     break
    ### torch model, param, save



# #############################################################################
#
# for data in trainer.data_loader:
#     break
#
# encoder_input = data["encoder"][:, 1:].to(trainer.device)  # 99
# decoder_input = data["decoder"].to(trainer.device)  # 100######################
#
# loss = model(encoder_input, decoder_input)
#
# torch.argmax(model.logits[0], -1)
#
# decoder_input[0]
# y = decoder_input
# dec = y * (1 - y.eq(2.).float())
# dec = dec[:, :-1].long()
# dec[0]
#
#
# label = y[:, 1:].contiguous()
# ########################################################################
# print('train finished...')
# # test evaluation
#
# print("finished !! ")

    



