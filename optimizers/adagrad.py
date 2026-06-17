import numpy as np

from optimizers.optimizer_base import OptimizerBase


class AdaGrad(OptimizerBase):
    def __init__(self, grad, x0, lr=0.1, eps_ad=1e-8, **kwargs):
        super().__init__(grad, x0, **kwargs)

        self.lr = lr
        self.eps_ad = eps_ad
        self.g2 = np.zeros_like(self.x0)

    def step(self, x):
        g = self.grad(x)
        self.g2 += g ** 2
        return x - self.lr * g / (np.sqrt(self.g2) + self.eps_ad)

    def __str__(self):
        return self.__class__.__name__ + f" (lr={self.lr})"
