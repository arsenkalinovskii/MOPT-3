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

    method_dir = os.path.join(func_dir, 'Momentum')
    ensure_dir(method_dir)

    # Моментум
    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    configs = [
        {'lr': lr, 'beta': 0.9}
        for lr in lr_values
    ]

    save_parameter_canvas(
        func,
        Momentum,
        configs,
        os.path.join(method_dir, 'lr_paths.png')
    )

    beta_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]

    configs = [
        {'lr': 0.01, 'beta': beta}
        for beta in beta_values
    ]

    save_parameter_canvas(
        func,
        Momentum,
        configs,
        os.path.join(method_dir, 'beta_paths.png')
    )

    save_heatmap_iterations(
        Momentum,
        func,
        'lr',
        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'beta',
        [0.5, 0.7, 0.8, 0.9, 0.99],
        os.path.join(method_dir, 'heatmap.png')
    )

    # Нестеров
    method_dir = os.path.join(func_dir, 'Nesterov')
    ensure_dir(method_dir)

    configs = [
        {'lr': lr, 'beta': 0.9}
        for lr in lr_values
    ]

    save_parameter_canvas(
        func,
        Nesterov,
        configs,
        os.path.join(method_dir, 'lr_paths.png')
    )

    configs = [
        {'lr': 0.01, 'beta': beta}
        for beta in beta_values
    ]

    save_parameter_canvas(
        func,
        Nesterov,
        configs,
        os.path.join(method_dir, 'beta_paths.png')
    )

    save_heatmap_iterations(
        Nesterov,
        func,
        'lr',
        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'beta',
        [0.5, 0.7, 0.8, 0.9, 0.99],
        os.path.join(method_dir, 'heatmap.png')
    )

    # адаград
    method_dir = os.path.join(func_dir, 'AdaGrad')
    ensure_dir(method_dir)

    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    configs = [
        {'lr': lr}
        for lr in lr_values
    ]

    save_parameter_canvas(
        func,
        AdaGrad,
        configs,
        os.path.join(method_dir, 'lr_paths.png')
    )

    save_line_iterations(
        AdaGrad,
        func,
        'lr',
        lr_values,
        {},
        method_dir
    )

    # RMSProp
    method_dir = os.path.join(func_dir, 'RMSProp')
    ensure_dir(method_dir)

    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    configs = [
        {'lr': lr, 'rho': 0.9}
        for lr in lr_values
    ]

    save_parameter_canvas(
        func,
        RMSProp,
        configs,
        os.path.join(method_dir, 'lr_paths.png')
    )

    rho_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]

    configs = [
        {'lr': 0.01, 'rho': rho}
        for rho in rho_values
    ]

    save_parameter_canvas(
        func,
        RMSProp,
        configs,
        os.path.join(method_dir, 'rho_paths.png')
    )

    save_heatmap_iterations(
        RMSProp,
        func,
        'lr',
        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'rho',
        [0.5, 0.7, 0.8, 0.9, 0.99],
        os.path.join(method_dir, 'heatmap.png')
    )

    # AdaDelta
    method_dir = os.path.join(func_dir, 'AdaDelta')
    ensure_dir(method_dir)

    rho_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]

    configs = [
        {'rho': rho}
        for rho in rho_values
    ]

    save_parameter_canvas(
        func,
        AdaDelta,
        configs,
        os.path.join(method_dir, 'rho_paths.png')
    )

    save_line_iterations(
        AdaDelta,
        func,
        'rho',
        rho_values,
        {},
        method_dir
    )

    # Adam
    method_dir = os.path.join(func_dir, 'Adam')
    ensure_dir(method_dir)

    beta1_values = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]

    configs = [
        {
            'lr': 0.01,
            'beta1': beta1,
            'beta2': 0.999
        }
        for beta1 in beta1_values
    ]

    save_parameter_canvas(
        func,
        Adam,
        configs,
        os.path.join(method_dir, 'beta1_paths.png')
    )

    beta2_values = [0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

    configs = [
        {
            'lr': 0.01,
            'beta1': 0.9,
            'beta2': beta2
        }
        for beta2 in beta2_values
    ]

    save_parameter_canvas(
        func,
        Adam,
        configs,
        os.path.join(method_dir, 'beta2_paths.png')
    )

    save_heatmap_iterations(
        Adam,
        func,
        'beta1',
        [0.5, 0.7, 0.8, 0.9, 0.99],
        'beta2',
        [0.9, 0.95, 0.99, 0.995, 0.999],
        os.path.join(method_dir, 'heatmap.png')
    )


def main():
    task1()
    # task2()
    # task3()
    # task4()


if __name__ == '__main__':
    main()
