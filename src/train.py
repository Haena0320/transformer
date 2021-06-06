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
        self.train_loss = list()
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

    def train_epoch(self, model, epoch, save_path=None, sp=None, md=None):
        if self.type == "train":
            model.train()

        else:
            model.eval()
            total_bleu = list()

        model.to(self.device)


        for data in tqdm(self.data_loader, desc="Epoch : {}".format(epoch)):
            #with amp.autocast():
            encoder_input = data["encoder"].to(self.device)  # 16, 100
            decoder_input = data["decoder"].to(self.device)  # 16, 100

            loss = model(encoder_input, decoder_input)
            # model(encoder_input, decoder_input)
            if self.type == 'train':
                if self.global_step % self.ckpnt_step == 0:
                    torch.save({"epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "lr_step":self.scheduler._step},
                               save_path + "/ckpnt_{}".format(epoch))

                self.optim_process(model, self.step, loss)
                self.step += 1


            else:
                sos = decoder_input[:, 0]  # bs
                sos = sos.unsqueeze(1)  # (bs, 1)

                bs, max_sent_len = decoder_input.size()
                max_sent_len += 50

                pred_token = torch.zeros(bs, max_sent_len)
                for i in range(max_sent_len):
                    y = model.search(encoder_input, sos)  # bs,
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
                    decode_truth = sp.DecodeIds(decoder_input[i, :].tolist())
                    print(decode_tokens)
                    print(decode_truth)
                    pred = md.detokenize(decode_tokens.strip().split())
                    truth = md.detokenize(decode_truth.strip().split())
                    bleu = sacrebleu.corpus_bleu(pred, truth)
                    total_bleu.append(bleu.score)
                    print("bleu {}".format(bleu.score))

        if self.type != "train":
            print("total_bleu per epoch : {}".format(sum(total_bleu) / len(total_bleu)))
        else:
            return None


    def optim_process(self, model, optim_step, loss):
        loss /= self.accum
        #self.gradscaler.scale(loss).backward()
        loss.backward()
        if optim_step % self.accum == 0:
            #self.gradscaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
            # self.gradscaler.step(self.optimizer)
            # self.gradscaler.update()
            self.scheduler.step()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.log_writer(loss.data*self.accum, self.global_step)
            self.global_step += 1