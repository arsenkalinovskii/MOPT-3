import numpy as np

from optimizers.optimizer_base import OptimizerBase


class Adam(OptimizerBase):
    def __init__(self, grad, x0, lr=0.01, beta1=0.9, beta2=0.999, eps_ad=1e-8, **kwargs):
        super().__init__(grad, x0, **kwargs)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_ad = eps_ad

        self.m = np.zeros_like(self.x0)
        self.v = np.zeros_like(self.x0)

        self.t = 0

    def step(self, x):
        self.t += 1

        g = self.grad(x)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps_ad)
