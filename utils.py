import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from functions import *
from optimizers import *

EPS = 1e-8
MAX_ITER = 100000
X0_QUAD = np.array([5.0, 5.0])


def run_optimizer(method, func, x0, **params):
    try:
        opt = method(grad=func.grad_f, x0=x0, eps=EPS, max_iter=MAX_ITER, **params)
        x, path, iters = opt.optimize()

        if np.any(np.isnan(x)):
            return None
        if np.any(np.isinf(x)):
            return None

        return {'x': x, 'path': path, 'iters': iters, 'value': func.f(x)}

    except Exception:
        return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
