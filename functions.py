import numpy as np


class func():
    def __init__(self, f, grad_f, name):
        self.f = f
        self.grad_f = grad_f
        self.name = name

    def __str__(self):
        return self.name


def f_func(x):
    return x[0] ** 2 + x[1] ** 2


def grad_f(x):
    return np.array([2 * x[0], 2 * x[1]])


f = func(f_func, grad_f, 'Good function')


def g_func(x):
    return x[0] ** 2 + 100 * x[1] ** 2


def grad_g(x):
    return np.array([2 * x[0], 200 * x[1]])


g = func(g_func, grad_g, 'Bad function')


def rosenbrock_func(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


rosenbrock = func(rosenbrock_func, grad_rosenbrock, 'Rosenbrock function')


def himmelblau_func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def grad_himmelblau(x):
    dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx, dy])


himmelblau = func(himmelblau_func, grad_himmelblau, 'Himmelblau function')


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


ackley = func(ackley_func, grad_ackley, 'Ackley function')
