import torch
import os
from torch.utils.data import DataLoader, Dataset
from src.model import *
import torch.optim as optim
import time
from tqdm import tqdm

## train_loader
def get_trainer(config, args,device, data_loader, writer, type):
    return Trainer(config, args,device, data_loader, writer, type)

def get_optimizer(model, args_optim):
    if args_optim =="adam":
        return torch.optim.Adam(model.parameters(), betas=(0.9, 0.98),eps=1e-09)

def get_lr_schedular(optimizer, config, total_steps):
    return WarmupLinearScheduler(optimizer, config, total_steps)


class WarmupLinearSchedular(object):
    def __init__(self, optimizer, config, total_steps):
        self.warmup_steps = config.train.warmup_steps
        self.max_lr = config.train.max_lr
        self.d_model = config.model.h_units
        self.total_steps = total_steps
        self.optimizer = optimizer

    def adjust_lr(self, total_steps):
        lr = self.d_model**(-0.5)*min(total_steps**(-0.5), total_steps*self.warmup_steps**(-1.5))
        for g in self.optimizer.param_groups:
            g["lr"] = lr



class Trainer:
    def __init__(self, config, args, device, data_loader, writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.writer = writer
        self.type = type

        self.global_step = 0

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_schedular(self, scheduler):
        self.sceduler = scheduler

    def log_writer(self, log, step):
        if self.type =="train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train_loss/step", log, self.global_step)
            self.writer.add_scalar("lr/step",lr, self.global_step)
        else:
            self.writer.add_scalar("valid_loss/loss", log, step)
        self.writer.flush()
        

    def train_epoch(self, model, epoch, global_step=None):
        if self.type =="train":
            model.train()

        else:
            model.eval()

        model.to(self.device)
        loss_save = list()

        for data in tqdm(self.data_loader, desc="Epoch : {}".format(epoch)):
            encoder_input = data["encoder"].to(self.device)
            decoder_input = data["decoder"].to(self.device)

            loss,  acc = model.forward(encoder_input, decoder_input)
            
            if self.type =='train':
                self.optim_process(model, loss)
                self.global_step+=1
                self.log_writer(loss, self.global_step)
            
            else:
                ## predicted value visualization
                loss_save.append(loss.item())
                
        if self.type != "train":
            loss = sum(loss_save)/len(loss_save)
            self.log_writer(loss, global_step)
            return loss

    def optim_process(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), self.config.train.clip)
        self.optimizer.step()
        self.schedular.step(self.global_step)
    
        






