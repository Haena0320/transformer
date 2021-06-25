import torch
import os
from torch.utils.data import DataLoader, Dataset
from src.model_2 import *
import torch.optim as optim
import time
from tqdm import tqdm
from torch.cuda import amp
from sacremoses import MosesDetokenizer
import sacrebleu
import torch.nn as nn


## train_loader
def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)

def get_optimizer(model, args_optim):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09)
    if args_optim == 'adamW':
        return torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09, weight_decay=0.01, amsgrad=False)


def get_lr_schedular(optimizer, config):
    h_units = config.model.h_units
    warmup = config.train.warmup
    return WarmupLinearshedular(optimizer, h_units, warmup)


class WarmupLinearshedular:
    def __init__(self, optimizer, h_units, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.h_units = h_units
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        self.optimizer.param_groups[0]["lr"] = rate
        self._rate = rate

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.h_units ** (-.5) * min(step ** (-.5), step * self.warmup ** (-1.5))


class Trainer:
    def __init__(self, config, args, device, data_loader, writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.writer = writer
        self.type = type
        self.accum = self.config.train.accumulation_step
        self.ckpnt_step = self.config.train.ckpnt_step
        self.gradscaler = amp.GradScaler()
        self.global_step = 1
        self.step = 0
        self.train_loss = 0
        #self.loss_fn = LabelSmoothingLoss(label_smoothing=0.1,tgt_vocab_size=37000, device=device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_schedular(self, scheduler):
        self.scheduler = scheduler

    def log_writer(self, log, step):
        if self.type == "train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", log, self.global_step)
            self.writer.add_scalar("train/lr", lr, self.global_step)
        else:
            self.writer.add_scalar("valid/loss", log, step)

    def train_epoch(self, model, epoch, save_path=None, sp=None, md=None):
        if self.type == "train":
            model.train()

        else:
            model.eval()
            total_bleu = list()

        model.to(self.device)


        for data in tqdm(self.data_loader, desc="Epoch : {}".format(epoch), ncols = 100):
            with amp.autocast():
                encoder_input = data["encoder"].to(self.device)  # 16, 100
                decoder_input = data["decoder"][:, :-1].to(self.device)  # 16, 99
                decoder_label = data["decoder"][:, 1:].to(self.device) # 16, 99
                output = model(encoder_input, decoder_input)
                B, S = decoder_label.size()
                loss = self.loss_fn(output.view(B*S, -1), decoder_label.view(-1))

                if self.type == 'train':
                    if self.global_step % self.ckpnt_step == 0:
                        torch.save({"epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "lr_step":self.scheduler._step},
                                save_path + "/ckpnt_{}".format(epoch))

                    self.optim_process(model, self.step, loss)
                    self.step += 1

        if self.type != "train":
            print("total_bleu per epoch : {}".format(sum(total_bleu) / len(total_bleu)))
        else:
            return None


    def optim_process(self, model, optim_step, loss):
        self.train_loss += loss.item() / self.accum
        loss /= self.accum
        self.gradscaler.scale(loss).backward()

        if optim_step % self.accum == 0:
            self.gradscaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)

            self.scheduler.step()
            self.gradscaler.step(self.optimizer)
            self.gradscaler.update()
            self.optimizer.zero_grad()
            self.log_writer(self.train_loss, self.global_step)

            self.train_loss = 0
            self.global_step += 1