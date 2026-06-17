import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from functions import *
from optimizers import *

EPS = 1e-8
MAX_ITER = 100000


def run_optimizer(method, func, x0, **params):
    try:
        opt = method(grad=func.grad_f, x0=x0, eps=EPS, max_iter=MAX_ITER, **params)
        path, iters = opt.optimize()

        path = np.array([i for i in path if not (np.any(np.isnan(i)) or np.any(np.isnan(i)))])
        if len(path) == 0:
            return None

        x = path[-1]
        return {'x': x, 'path': path, 'iters': iters, 'value': func.f(x)}

    except Exception as e:
        print(method, func, ' - Optimization failed:', e)
        return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
