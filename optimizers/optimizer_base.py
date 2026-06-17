import numpy as np


class OptimizerBase:
    def __init__(self, grad, x0, eps=1e-6, max_iter=10000):
        self.grad = grad
        self.x0 = np.array(x0, dtype=float)
        self.eps = eps
        self.max_iter = max_iter

    def step(self, x):
        raise NotImplementedError

    def optimize(self):
        x = self.x0.copy()
        path = [x.copy()]

        for k in range(self.max_iter):
            g = self.grad(x)

            if np.linalg.norm(g) < self.eps:
                break

            x = self.step(x)
            path.append(x.copy())

        return np.array(path), k + 1
