# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.optim import SGD
from torch import nn
import os
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--arch', type=str, default="resnet")
    parser.add_argument('--use_lr_scheduler', type=int, default=0)
    parser.add_argument('--lr_steps', type=int, nargs='*', default=[70, 90])
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.get_optimizer()

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_epoch(self, dataset, epoch) -> None:
        if self.scheduler is not None:
            if self.args.arch == "vit":
                self.scheduler.step(epoch)
            else:
                self.scheduler.step()

    def end_task(self, dataset) -> None:
        # reset optimizer
        self.get_optimizer()

    def get_optimizer(self):
        # reset optimizer
        if self.args.arch == "vit":
            self.opt = create_optimizer_v2(self.net, opt='adamw', lr=1e-3, weight_decay=0.05)
        else:
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        if self.args.use_lr_scheduler:
            # Pass through Buffer
            if self.args.arch == "vit":
                self.scheduler, _ = create_scheduler({"sched": "cosine", "epochs": self.args.n_epochs}, self.opt)
            else:
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None
