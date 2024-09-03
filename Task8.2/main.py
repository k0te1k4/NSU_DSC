import math

from scipy.integrate import tplquad


def f(x, y, z):
    return (x + y + z) / math.sqrt(2 * x * x + 4 * y * y + 5 * z * z)


def y_upper_limit(x):
    return math.sqrt(1 - x * x)


def z_upper_limit(x, y):
    return math.sqrt(1 - x * x - y * y)


print(tplquad(f, 0, 1, 0, y_upper_limit, 0, z_upper_limit))
