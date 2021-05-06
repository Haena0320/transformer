import torch
import os
from torch.utils.data import DataLoader, Dataset
from src.model_2 import *
import torch.optim as optim
import time
from tqdm import tqdm


## train_loader
def get_trainer(config, args, device, data_loader, writer, type):
    return Trainer(config, args, device, data_loader, writer, type)


def get_optimizer(model, args_optim):
    if args_optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09)
    if args_optim == 'adamW':
        return torch.optim.AdamW(params, lr=0, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)


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


    def train_epoch(self, model, epoch, global_step=None, save_path=None):
        if self.type == "train":
            model.train()

        else:
            model.eval()

        model.to(self.device)
        loss_save = list()

        for data in tqdm(self.data_loader, desc="Epoch : {}".format(epoch)):
            encoder_input = data["encoder"][:, 1:].to(self.device)  # 16, 99
            decoder_input = data["decoder"].to(self.device)  # 16, 100
            loss = model(encoder_input, decoder_input)
            # model(encoder_input, decoder_input)
            if self.type == 'train':

                if self.global_step % self.ckpnt_step == 0:
                    torch.save({"epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_stata_dict": self.optimizer.state_dict()}, save_path+"ckpnt_{}".format(epoch))

                self.optim_process(model, self.global_step, loss)
                self.global_step += 1
            else:
                ## predicted value visualization
                ## need check
                bs, input_length = encoder_input.size()
                decode_token = torch.ones(bs).unsqueeze(1) # (bs, 1) -> <bos> token id = 1
                max_length = input_length + 50
                for i in range(len(max_length)):
                    decode_output = model.search(encoder_input, decode_token)
                    print(decode_output)
                    print("decode output size {}".format(decode_output.size()))
                    decode_token = torch.cat((decode_token, decode_output),dim=-1)
                    print("decode token size {}".format(decode_token.size()))
                    print(decode_token)

                loss_save.append(loss.item())

        if self.type != "train":
            loss = sum(loss_save) / len(loss_save)
            self.log_writer(loss, global_step)
            return loss

    def optim_process(self, model, optim_step, loss):
        loss /= self.accum
        loss.backward()
        self.train_loss += loss.data
        if optim_step % self.accum == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.log_writer(self.train_loss, self.global_step)
            self.train_loss = 0
            
            
            

# acummulation_step = 25
# batch_size = 50
# step = 0
#
# loss = model ( data, label)
# loss /= acummulation_step
# loss.backward()
# step += 1
# if step == acummulation_step:
#     optimizer.step()
#     step = 0
#     optimizer.zero_grad()
