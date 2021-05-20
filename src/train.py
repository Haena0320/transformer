import torch
import os
from torch.utils.data import DataLoader, Dataset
from src.model_2 import *
import torch.optim as optim
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
from sacremoses import MosesDetokenizer
import sacrebleu


## train_loader
def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)


def get_optimizer(model, args_optim):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09)
    if args_optim == 'adamW':
        return torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)


def get_lr_schedular(optimizer, config):
    h_units = config.model.h_units
    warmup = config.train.warmup
    # return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=4000, step_size_down=total_iter_num,cycle_momentum=False, mode='triangular')
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
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

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
        self.global_step = 1
        self.train_loss = 0
        if self.type != "train":
            self.md = MosesDetokenizer(lang="du")

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

    def train_epoch(self, model, epoch, save_path=None, sp=None):
        if self.type == "train":
            model.train()

        else:
            model.eval()
            total_bleu = list()

        model.to(self.device)


        for data in tqdm(self.data_loader, desc="Epoch : {}".format(epoch)):
            with autocast():
                encoder_input = data["encoder"][:, 1:].to(self.device)  # 16, 99
                decoder_input = data["decoder"].to(self.device)  # 16, 100
                loss = model(encoder_input, decoder_input)
                # model(encoder_input, decoder_input)
                if self.type == 'train':
                    if self.global_step % self.ckpnt_step == 0:
                        torch.save({"epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_stata_dict": self.optimizer.state_dict()},
                                   save_path + "/ckpnt_{}".format(epoch))

                    self.log_writer(loss.data, self.global_step)
                    self.optim_process(model, self.global_step, loss)
                    self.global_step += 1

                else:
                    bs, input_length = encoder_input.size()
                    sos = torch.ones(bs, dtype=int, device="cuda:0").unsqueeze(1)  # (bs, 1) -> <bos> token id = 1
                    max_length = input_length + 50

                    for i in range(max_length):
                        y = model.search(encoder_input, sos)
                        sos = torch.cat([sos, y], dim=-1)

                    pred = list()
                    truth = list()

                    tokens = sos[:,1:].tolist()
                    for i, token in enumerate(tokens):
                        for j in range(len(token)):
                            if token[j] == 2:
                                token = token[:j]
                                break
                        de_token = sp.DecodeIds(token)
                        de_truth = sp.DecodeIds(decoder_input[i,:].tolist())
                        pred.append(de_token)
                        truth.append(de_truth)

                    pred = [self.md.detokenize(i.strip().split()) for i in pred]
                    truth = [self.md.detokenize(i.strip().split()) for i in truth]

                    assert len(pred) == len(truth)
                    bleu = [sacrebleu.corpus_bleu(pred[i], truth[i]).score for i in range(len(pred))]
                    total_bleu.append(sum(bleu) / len(bleu))
        if self.type != "train":
            print("total_bleu per epoch : {}".format(sum(total_bleu) / len(total_bleu)))




    def optim_process(self, model, optim_step, loss):
        loss /= self.accum
        loss.backward()
        self.train_loss += loss.data
        if optim_step % self.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.train_loss = 0