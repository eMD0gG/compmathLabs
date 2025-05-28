from math import cos, sin, factorial
import numpy as np
from typing import Callable, Optional, Any
from interpolation_module import find_n_closest


def differentiation_lagrange(x_values: list[float], y_values: list[float], n: int, x_diff: float) -> float:
    """Вычисляет производную 1-го порядка в точке x_diff с использованием интерполяционного многочлена Лагранжа.

      Args:
          x_values: Список значений аргумента функции.
          y_values: Список значений функции, соответствующих x_values.
          n: Количество точек интерполяции.
          x_diff: Точка, в которой вычисляется производная.

      Returns:
          Значение 1-й производной в точке x_diff, вычисленное с помощью интерполяционного многочлена Лагранжа.
    """
    indexes = find_n_closest(x_diff, x_values, n)
    x_values = x_values[indexes[0]:indexes[-1] + 1]
    y_values = y_values[indexes[0]:indexes[-1] + 1]
    h = x_values[1] - x_values[0]
    x_values = np.array(x_values)

    result = 0

    for i in range(n):
        tmp1 = 0
        tmp2 = 1
        for j in range(n):
            if j != i:
                mask = (np.arange(len(x_values)) != i) & (np.arange(len(x_values)) != j)
                tmp1 += np.prod(x_diff - x_values[mask])
                tmp2 *= (i - j)

        result += tmp1 * y_values[i] / tmp2

    result /= h ** (n - 1)

    return result



func = lambda x: x ** 3 - cos(2 * x)
der = lambda x: 3 * x ** 2 + 2*sin(2*x)

interval = [0.1, 0.6]

n = 11

grid = np.linspace(interval[0], interval[1], n)

x_values = list(grid)
y_values = list(map(func, x_values))

print(differentiation_lagrange(x_values, y_values, 7, 0.25), der(0.25), differentiation_lagrange(x_values, y_values, 7, 0.25) - der(0.25))


max_der = 128*sin(0.8)
fctrl = factorial(7)
w = 0

for i in range(n):
    x_values = np.array(x_values[:7])
    mask = (np.arange(len(x_values)) != i)
    w += np.prod(x_values[3] - x_values[mask])

min_R = 0
max_R = max_der * w /fctrl

print(min_R, max_R)

