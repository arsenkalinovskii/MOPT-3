import numpy as np

from optimizers.optimizer_base import OptimizerBase


class Momentum(OptimizerBase):
    def __init__(self, grad, x0, lr=0.01, beta=0.9, **kwargs):
        super().__init__(grad, x0, **kwargs)

        self.lr = lr
        self.beta = beta
        self.m = np.zeros_like(self.x0)

    def step(self, x):
        g = self.grad(x)
        self.m = self.beta * self.m + g
        return x - self.lr * self.m
