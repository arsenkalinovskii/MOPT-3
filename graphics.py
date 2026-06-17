import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from functions import *
from optimizers import *
from utils import *


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

    sns.heatmap(
        df,
        annot=True,
        fmt='.0f',
        cmap='viridis'
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


def save_lineplot(x, y, xlabel, title, filename):
    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        '-o',
        color='red',
        markersize=5,
        linewidth=1.5
    )

    plt.xlabel(xlabel)
    plt.ylabel('Function value f(x)')

    plt.title(title)

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    if np.min(y) > 0:
        if np.max(y) / np.min(y) > 100:
            plt.yscale('log')
            plt.ylabel('Function value f(x) (log scale)')
        elif np.max(y) - np.min(y) < 1e-6:
            plt.ylim(np.min(y) - 0.1, np.max(y) + 0.1)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


def draw_path(func, path, title, filename):
    left, right = func.range

    x = np.linspace(left, right, 200)
    y = np.linspace(left, right, 200)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func.f(np.array([X[i, j], Y[i, j]]))

    path_x = np.nan_to_num(path[:, 0], nan=np.nan, posinf=np.nan, neginf=np.nan)
    path_y = np.nan_to_num(path[:, 1], nan=np.nan, posinf=np.nan, neginf=np.nan)

    contours = plt.contour(X, Y, Z, levels=15, cmap='viridis', linewidths=0.8)

    plt.clabel(contours, inline=True, fontsize=7, fmt='%1.1f')

    plt.plot(path_x, path_y, '-o', color='red', markersize=3)

    plt.plot(path_x[0], path_y[0], 'go', markersize=8)

    if not np.isnan(path_x[-1]) and not np.isnan(path_y[-1]):
        plt.plot(path_x[-1], path_y[-1], 'rX', markersize=10)

    for minimum in func.mins:
        plt.plot(minimum[0], minimum[1], '*', markersize=12, color='gold')

    plt.xlim(left, right)
    plt.ylim(left, right)

    plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.grid(True, linestyle='--', alpha=0.5)


def save_parameter_canvas(method, func, x0_quad, configs, filename):
    plt.figure(figsize=(14, 8))

    for i, params in enumerate(configs):
        result = run_optimizer(method, func, x0_quad, **params)

        plt.subplot(2, 3, i + 1)

        if result is not None:
            draw_path(func, result['path'], str(params), None)
        else:
            plt.title('Diverged')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_heatmap_iterations(method, func, x0_quad, p1_name, p1_values, p2_name, p2_values, filename):
    table = []

    for p1 in p1_values:
        row = []
        for p2 in p2_values:
            result = run_optimizer(method, func, x0_quad, **{p1_name: p1, p2_name: p2})
            if result is None:
                row.append(MAX_ITER)
            else:
                row.append(result['iters'])

        table.append(row)

    df = pd.DataFrame(table, index=p1_values, columns=p2_values)
    df.to_csv(filename.replace('.png', '.csv'))

    save_heatmap(df, f'{func} iterations', filename)


def save_line_iterations(method, func, x0_quad, p_name, values, params, dirname):
    rows = []

    for value in values:
        cur_params = params.copy()
        cur_params[p_name] = value

        result = run_optimizer(method, func, x0_quad, **cur_params)

        if result is None:
            iters = MAX_ITER
        else:
            iters = result['iters']

        rows.append([value, iters])

    df = pd.DataFrame(rows, columns=[p_name, 'iterations'])
    df.to_csv(os.path.join(dirname, 'table.csv'), index=False)

    save_lineplot(
        df[p_name],
        df['iterations'],
        p_name,
        f'{func}',
        os.path.join(dirname, 'lineplot.png')
    )
