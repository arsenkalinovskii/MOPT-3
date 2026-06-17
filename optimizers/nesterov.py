import numpy as np

from optimizers.optimizer_base import OptimizerBase


class Nesterov(OptimizerBase):
    def __init__(self, grad, x0, lr=0.01, beta=0.9, **kwargs):
        super().__init__(grad, x0, **kwargs)

        self.lr = lr
        self.beta = beta
        self.v = np.zeros_like(self.x0)

    def step(self, x):
        y = x - self.beta * self.v
        g = self.grad(y)
        self.v = self.beta * self.v + self.lr * g
        return x - self.v

    def __str__(self):
        return self.__class__.__name__ + f" (lr={self.lr}, beta={self.beta})"
