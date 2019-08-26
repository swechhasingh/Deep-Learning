import numpy as np
import torch.optim as optim
import math


class CustomOptimizer:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.lr = 0

    def step(self):
        self.step_num += 1
        lrate = self.rate()
        self.lr = lrate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lrate
        self.optimizer.step()

    def rate(self, step=None):

        if step is None:
            step = self.step_num
        lrate = (self.d_model) ** (-0.5) * min(
            1.0 / math.sqrt(step), step * math.pow(self.warmup_steps, -1.5)
        )
        return lrate

    def des_opt(self):
        des = {"step_num": self.step_num, "lr": self.lr}
        return des

