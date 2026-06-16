import numpy as np


def quad_good(x):
    return x[0] ** 2 + x[1] ** 2


def quad_good_grad(x):
    return np.array([2 * x[0], 2 * x[1]])


def quad_bad(x):
    return x[0] ** 2 + 100 * x[1] ** 2


def quad_bad_grad(x):
    return np.array([2 * x[0], 200 * x[1]])


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def himmelblau(x):
    a = x[0] ** 2 + x[1] - 11
    b = x[0] + x[1] ** 2 - 7
    return a ** 2 + b ** 2


def himmelblau_grad(x):
    a = x[0] ** 2 + x[1] - 11
    b = x[0] + x[1] ** 2 - 7
    return np.array([4 * x[0] * a + 2 * b, 2 * a + 4 * x[1] * b])
