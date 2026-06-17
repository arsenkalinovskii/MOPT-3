import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class func():
    def __init__(self, f, grad_f, name, func_min, func_range=(-4, 4)):
        self.f = f
        self.grad_f = grad_f
        self.name = name
        self.range = func_range
        self.mins = func_min

    def __str__(self):
        return self.name


def f_func(x):
    return x[0] ** 2 + x[1] ** 2


def grad_f(x):
    return np.array([2 * x[0], 2 * x[1]])


f = func(f_func, grad_f, 'Good', [(0, 0)], (-4, 4))


def g_func(x):
    return x[0] ** 2 + 100 * x[1] ** 2


def grad_g(x):
    return np.array([2 * x[0], 200 * x[1]])


g = func(g_func, grad_g, 'Bad', [(0, 0)], (-2, 2))


def rosenbrock_func(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


rosenbrock = func(rosenbrock_func, grad_rosenbrock, 'Rosenbrock', [(1, 1)], (-4, 4))


def himmelblau_func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def grad_himmelblau(x):
    dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx, dy])


himmelblau = func(himmelblau_func, grad_himmelblau, 'Himmelblau',
                  [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)], (-6, 6))


def ackley_func(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return term1 + term2 + np.e + 20


def grad_ackley(x):
    r_sq = x[0] ** 2 + x[1] ** 2
    # Защита от деления на ноль в точке глобального минимума (0,0)
    if r_sq < 1e-12:
        return np.array([0.0, 0.0])

    term1_base = np.exp(-0.2 * np.sqrt(0.5 * r_sq))
    term2_base = np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))

    dx = (2 * x[0] * term1_base) / np.sqrt(0.5 * r_sq) + np.pi * np.sin(2 * np.pi * x[0]) * term2_base
    dy = (2 * x[1] * term1_base) / np.sqrt(0.5 * r_sq) + np.pi * np.sin(2 * np.pi * x[1]) * term2_base

    return np.array([dx, dy])


ackley = func(ackley_func, grad_ackley, 'Ackley', [(0, 0)], [-4, 4])

if __name__ == '__main__':
    os.makedirs('funcs_demo', exist_ok=True)

    functions = [f, g, rosenbrock, himmelblau, ackley]

    for func in functions:
        fig = plt.figure(figsize=(14, 6))

        x = np.linspace(func.range[0], func.range[1], 100)
        y = np.linspace(func.range[0], func.range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func.f([X[i, j], Y[i, j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('x');
        ax1.set_ylabel('y');
        ax1.set_zlabel('f(x,y)')
        ax1.set_title(f'{func.name}: 3D')

        ax2 = fig.add_subplot(122)
        ax2.contour(X, Y, Z, levels=20, cmap='viridis')
        for m in func.mins:
            ax2.plot(m[0], m[1], 'r*', markersize=12)
        ax2.set_xlabel('x');
        ax2.set_ylabel('y')
        ax2.set_title(f'{func.name}: Contour')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'funcs_demo/{func.name}_demo.png', dpi=300)
        plt.close()

        print(f'Generated {func.name}')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, func in enumerate(functions):
        ax = axes.flat[idx]
        x = np.linspace(func.range[0], func.range[1], 100)
        y = np.linspace(func.range[0], func.range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func.f([X[i, j], Y[i, j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])

        ax.contour(X, Y, Z, levels=15, cmap='viridis')
        for m in func.mins:
            ax.plot(m[0], m[1], 'r*', markersize=10)
        ax.set_title(func.name)
        ax.grid(True, alpha=0.3)

    axes.flat[5].set_visible(False)
    plt.tight_layout()
    plt.savefig('funcs_demo/comparison.png', dpi=300)
    plt.close()

    print('All plots generated!')
