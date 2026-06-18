from utils import *
from functions import *

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def save_heatmap(df, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.0f', cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_lineplot(x, y, xlabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, '-o', color='red', markersize=5)
    plt.xlabel(xlabel)
    plt.yscale('log')
    plt.ylabel('Iterations')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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


def save_parameter_canvas(method, func: Func, x0_quad, configs, filename):
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
            if result is not None:
                row.append(result['iters'])
            else:
                row.append(MAX_ITER)
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

        if result is not None:
            iters = result['iters']
        else:
            iters = MAX_ITER
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


def build_all_methods_plot(func, x0, params, filename):
    left, right = func.range

    x = np.linspace(left, right, 200)
    y = np.linspace(left, right, 200)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func.f(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))
    contours = plt.contour(X, Y, Z, levels=25, cmap='viridis', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=7, fmt='%1.1f')

    for idx, (method_name, method) in enumerate(METHODS.items()):
        result = run_optimizer(method, func, x0[idx % len(x0)], **params[method_name])
        if result is None:
            continue

        path = result['path']
        plt.plot(path[:, 0], path[:, 1], '-o', markersize=2, label=method_name)

    for i, min_point in enumerate(func.mins):
        min_point = np.array(min_point)
        plt.scatter(min_point[0], min_point[1],
                    s=120, marker='*', color='red',
                    edgecolors='black', linewidth=1.5,
                    label='min' if i == 0 else '')

    r = abs(func.range[0])
    plt.xlim(-r, r)
    plt.ylim(-r, r)

    plt.title(str(func))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
