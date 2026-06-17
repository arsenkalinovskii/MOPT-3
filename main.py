import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from graphics import *
from functions import *
from optimizers import *

METHODS = {
    'Momentum': Momentum,
    'Nesterov': Nesterov,
    'AdaGrad': AdaGrad,
    'RMSProp': RMSProp,
    'AdaDelta': AdaDelta,
    'Adam': Adam
}


def task1():
    for func in [f, g]:
        func_dir = os.path.join(
            'task1',
            func.name.replace(' ', '_')
        )
        ensure_dir(func_dir)

        # task1_momentum(func, func_dir)
        # task1_nesterov(func, func_dir)
        # task1_ada_grad(func, func_dir)
        # task1_rms_prop(func, func_dir)
        # task1_ada_delta(func, func_dir)
        task1_adam(func, func_dir)


def task1_momentum(func, func_dir):
    method_dir = os.path.join(func_dir, 'Momentum')
    ensure_dir(method_dir)

    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    configs = [
        {'lr': lr, 'beta': 0.9}
        for lr in lr_values
    ]
    x0_quad = (-4, 4)

    save_parameter_canvas(Momentum, func, x0_quad, configs, os.path.join(method_dir, 'lr_paths.png'))

    beta_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]
    configs = [
        {'lr': 0.01, 'beta': beta}
        for beta in beta_values
    ]

    save_parameter_canvas(Momentum, func, x0_quad, configs, os.path.join(method_dir, 'beta_paths.png'))
    save_heatmap_iterations(Momentum, func, x0_quad, 'lr', lr_values, 'beta', beta_values,
                            os.path.join(method_dir, 'heatmap.png'))


def task1_nesterov(func, func_dir):
    method_dir = os.path.join(func_dir, 'Nesterov')
    ensure_dir(method_dir)

    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    configs = [
        {'lr': lr, 'beta': 0.9}
        for lr in lr_values
    ]
    x0_quad = (-4, 4)
    beta_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]

    save_parameter_canvas(Nesterov, func, x0_quad, configs, os.path.join(method_dir, 'lr_paths.png'))

    configs = [
        {'lr': 0.01, 'beta': beta}
        for beta in beta_values
    ]

    save_parameter_canvas(Nesterov, func, x0_quad, configs, os.path.join(method_dir, 'beta_paths.png'))
    save_heatmap_iterations(Nesterov, func, x0_quad, 'lr', lr_values, 'beta', beta_values,
                            os.path.join(method_dir, 'heatmap.png'))


def task1_ada_grad(func, func_dir):
    method_dir = os.path.join(func_dir, 'AdaGrad')
    ensure_dir(method_dir)

    lr_values = [0.01, 0.05, 0.1, 0.5, 1, 2]
    configs = [
        {'lr': lr}
        for lr in lr_values
    ]
    x0_quad = (-4, 4)

    save_parameter_canvas(AdaGrad, func, x0_quad, configs, os.path.join(method_dir, 'lr_paths.png'))
    save_line_iterations(AdaGrad, func, x0_quad, 'lr', lr_values, {}, method_dir)


def task1_rms_prop(func, func_dir):
    method_dir = os.path.join(func_dir, 'RMSProp')
    ensure_dir(method_dir)

    lr_values = [0.01, 0.05, 0.1, 0.5, 1, 2]
    configs = [
        {'lr': lr, 'rho': 0.9}
        for lr in lr_values
    ]
    x0_quad = (-4, 4)

    save_parameter_canvas(RMSProp, func, x0_quad, configs, os.path.join(method_dir, 'lr_paths.png'))
    rho_values = [0.1, 0.3, 0.6, 0.8, 0.9, 0.99]

    configs = [
        {'lr': 0.5, 'rho': rho}
        for rho in rho_values
    ]

    save_parameter_canvas(RMSProp, func, x0_quad, configs, os.path.join(method_dir, 'rho_paths.png'))
    save_heatmap_iterations(RMSProp, func, x0_quad, 'lr', lr_values, 'rho', rho_values,
                            os.path.join(method_dir, 'heatmap.png'))


def task1_ada_delta(func, func_dir):
    method_dir = os.path.join(func_dir, 'AdaDelta')
    ensure_dir(method_dir)

    rho_values = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
    configs = [
        {'rho': rho}
        for rho in rho_values
    ]
    x0_quad = (-4, 4)

    save_parameter_canvas(AdaDelta, func, x0_quad, configs, os.path.join(method_dir, 'rho_paths.png'))
    save_line_iterations(AdaDelta, func, x0_quad, 'rho', rho_values, {}, method_dir)


def task1_adam(func, func_dir):
    method_dir = os.path.join(func_dir, 'Adam')
    ensure_dir(method_dir)

    beta1_values = [0.00000001, 0.0001, 0.01, 0.1, 0.3, 0.8]
    x0_quad = (-4, 4)
    configs = [
        {
            'lr': 0.2,
            'beta1': beta1,
            'beta2': 0.9
        }
        for beta1 in beta1_values
    ]

    save_parameter_canvas(Adam, func, x0_quad, configs, os.path.join(method_dir, 'beta1_paths.png'))

    beta2_values = [0.8, 0.9, 0.95, 0.99, 0.999, 0.99999]
    configs = [
        {
            'lr': 1,
            'beta1': 0.9,
            'beta2': beta2
        }
        for beta2 in beta2_values
    ]

    save_parameter_canvas(Adam, func, x0_quad, configs, os.path.join(method_dir, 'beta2_paths.png'))
    save_heatmap_iterations(Adam,
                            func,
                            x0_quad,
                            'beta1',
                            beta1_values,
                            'beta2',
                            beta2_values,
                            os.path.join(method_dir, 'heatmap.png'))


def task2():
    for func in [f, g]:
        func_dir = os.path.join(
            'task2',
            func.name.replace(' ', '_')
        )
        ensure_dir(func_dir)

        task2_function(func, func_dir)


def task2_function(func, func_dir):
    x0_quad = (-4, 4)

    methods = [
        (Momentum, {'lr': 0.005, 'beta': 0.9}),
        (Nesterov, {'lr': 0.003, 'beta': 0.95}),
        (AdaGrad, {'lr': 0.8}),
        (RMSProp, {'lr': 0.5, 'rho': 0.9}),
        (AdaDelta, {'rho': 0.9999999}),
        (Adam, {'lr': 1, 'beta1': 0.9, 'beta2': 0.99})
    ]

    plt.figure(figsize=(14, 8))

    for i, (method, params) in enumerate(methods):
        name = method.__name__ + " " + str(params)

        plt.subplot(2, 3, i + 1)
        result = run_optimizer(method, func, x0_quad, **params)

        if result is not None:
            draw_path(func, result['path'], name, None)

    plt.tight_layout()
    plt.savefig(os.path.join(func_dir, 'all_methods.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rows = []

    for method, params in methods:
        name = method.__name__ + " " + str(params)

        result = run_optimizer(method, func, x0_quad, **params)

        if result is None:
            rows.append([name, MAX_ITER, np.nan])
        else:
            rows.append([name, result['iters'], result['value']])

    df = pd.DataFrame(rows, columns=['method', 'iterations', 'f(x)'])
    df.to_csv(os.path.join(func_dir, 'table.csv'), index=False)


if __name__ == '__main__':
    # task1()
    task2()
    # task3()
    # task4()
