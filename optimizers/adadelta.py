import numpy as np

from optimizers.optimizer_base import OptimizerBase


class AdaDelta(OptimizerBase):
    def __init__(self, grad, x0, rho=0.95, eps_ad=1e-6, **kwargs):
        super().__init__(grad, x0, **kwargs)

        self.rho = rho
        self.eps_ad = eps_ad
        self.eg = np.zeros_like(self.x0)
        self.ex = np.zeros_like(self.x0)

    def step(self, x):
        g = self.grad(x)
        self.eg = self.rho * self.eg + (1 - self.rho) * g ** 2
        dx = -np.sqrt(self.ex + self.eps_ad)
        dx /= np.sqrt(self.eg + self.eps_ad)
        dx *= g
        self.ex = self.rho * self.ex + (1 - self.rho) * dx ** 2

        return x + dx

    def __str__(self):
        return self.__class__.__name__ + f" (beta={self.rho})"
