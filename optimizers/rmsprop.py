import numpy as np

from optimizers.optimizer_base import OptimizerBase


class RMSProp(OptimizerBase):
    def __init__(self, grad, x0, lr=0.01, rho=0.9, eps_ad=1e-8, **kwargs):
        super().__init__(grad, x0, **kwargs)

        self.lr = lr
        self.rho = rho
        self.eps_ad = eps_ad
        self.g2 = np.zeros_like(self.x0)

    def step(self, x):
        g = self.grad(x)
        self.g2 = self.rho * self.g2 + (1 - self.rho) * g ** 2
        return x - self.lr * g / (np.sqrt(self.g2) + self.eps_ad)
