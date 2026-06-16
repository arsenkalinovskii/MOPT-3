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

START_POINTS = [
    np.array([-4.0, -4.0]),
    np.array([0.0, 0.0]),
    np.array([4.0, 4.0])
]

METHODS = {
    'Momentum': Momentum,
    'Nesterov': Nesterov,
    'AdaGrad': AdaGrad,
    'RMSProp': RMSProp,
    'AdaDelta': AdaDelta,
    'Adam': Adam
}

PARAMS = {
    'Momentum': {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'beta': [0.5, 0.7, 0.8, 0.9, 0.99]
    },
    'Nesterov': {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'beta': [0.5, 0.7, 0.8, 0.9, 0.99]
    },
    'AdaGrad': {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    },
    'RMSProp': {
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'rho': [0.5, 0.7, 0.8, 0.9, 0.99]
    },
    'AdaDelta': {
        'rho': [0.5, 0.7, 0.8, 0.9, 0.99]
    },
    'Adam': {
        'beta1': [0.5, 0.7, 0.8, 0.9, 0.99],
        'beta2': [0.9, 0.95, 0.99, 0.995, 0.999]
    }
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def save_heatmap(df, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_lineplot(x, y, xlabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel('Iterations')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def task1():
    base_dir = 'results/task1'

    ensure_dir(base_dir)
    quad_functions = [f, g]

    for func in quad_functions:
        func_dir = os.path.join(base_dir, func.name.replace(' ', '_'))
        ensure_dir(func_dir)

        for method_name, method in METHODS.items():
            method_dir = os.path.join(func_dir, method_name)

            ensure_dir(method_dir)
            params = PARAMS[method_name]

            if len(params) == 1:
                p_name = list(params.keys())[0]
                values = params[p_name]
                rows = []

                for value in values:
                    result = run_optimizer(method, func, X0_QUAD, **{p_name: value})
                    if result is None:
                        iters = MAX_ITER
                    else:
                        iters = result['iters']
                    rows.append([value, iters])

                df = pd.DataFrame(rows, columns=[p_name, 'iterations'])
                df.to_csv(os.path.join(method_dir, 'table.csv'), index=False)

                save_lineplot(df[p_name], df['iterations'], p_name, f'{method_name} - {func}',
                              os.path.join(method_dir, 'plot.png'))

            else:
                p1, p2 = list(params.keys())
                values1 = params[p1]
                values2 = params[p2]
                table = []

                for v1 in values1:
                    row = []
                    for v2 in values2:
                        result = run_optimizer(method, func, X0_QUAD, **{p1: v1, p2: v2})

                        if result is None:
                            row.append(MAX_ITER)
                        else:
                            row.append(result['iters'])
                    table.append(row)

                df = pd.DataFrame(table, index=values1, columns=values2)
                df.to_csv(os.path.join(method_dir, 'table.csv'))

                save_heatmap(df, f'{method_name} - {func}', os.path.join(method_dir, 'heatmap.png'))
