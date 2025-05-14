from math import *
import numpy as np
from typing import Callable, Optional, Any


def find_n_closest(target: float,
                           grid: list[float],
                           n: int) -> list[int]:
    """
    Ищет индексы n ближайших к target значений из сетки grid.
    Результат отсортирован по возрастанию значений.

    Args:
        target (float): Число, к которому ищутся ближайшие значения.
        grid (list[float]): Список чисел для поиска.
        n (int): Количество ближайших значений.

    Returns:
        list[int]: Отсортированный список индексов из n ближайших чисел.
    """
    indexed_grid = list(enumerate(grid))

    closest_indices = [index for index, value in sorted(
        indexed_grid,
        key=lambda x: abs(x[1] - target)
    )[:n]]

    closest_indices.sort(key=lambda x: grid[x])

    return closest_indices

def find_closest_index(values: list[float], target: float) -> int:
    """
        Ищет индекс ближайшего к target числа из values.

        Args:
            target (float): Число, к которому ищется ближайший индекс.
            values (list[float]): Список чисел для поиска.

        Returns:
           float: индекс ближайшего к target числа.
    """
    closest_index = min(range(n), key=lambda i: abs(values[i] - target))
    return closest_index


def divided_differences(x_points: list[float], y_points: list[float]) -> list[float]:
    """
    Вычисляет разделенные разности для интерполяционного многочлена Ньютона.

    Args:
        x_points (list): Список узлов интерполяции.
        y_points (list): Список значений функции в узлах.

    Retruns:
        list: Коэффициенты разделенных разностей.
    """
    n = len(x_points)

    table = [[0.0] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = y_points[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_points[i + j] - x_points[i])

    return table[0]


def lagrange_interpolate(x: float,
                         func: Callable[[float], float],
                         x_values: list[float],
                         y_values: list[float],
                         n: int = 11) -> float | None:
    """Вычисляет интерполяционного многочлена Лагранжа в точке x.

       Args:
           x (float): Точка, в которой вычисляется интерполированное значение.
           func (Callable[[float], float]): Функция, которую нужно интерполировать.
           x_values (list[float]): Узлы интерполяции.
           y_values (list[float]): Значения функции в узлах интерполяции.
           n (int, optional): Количество ближайших узлов для интерполяции.

       Returns:
           Optional[float]: Интерполированное значение или None.
     """
    indexes = np.array(find_n_closest(x, x_values, n))
    x_values = x_values[indexes[0]:indexes[-1] + 1]
    y_values = y_values[indexes[0]:indexes[-1] + 1]
    result = 0
    for i in range(n):
        j = np.arange(n) != i
        p = np.prod((x - x_values[j]) / (x_values[i] - x_values[j]))
        result += func(y_values[i]) * p

    return result


def newton_interpolate(x: float,
                       func: Callable[[float], float],
                       grid: list[float],
                       n: int = 11) -> float | None:
    """
    Вычисляет значение интерполяционного многочлена Ньютона в точке x.

    Args:
        x (float): Точка, в которой вычисляется интерполированное значение.
        func (Callable[[float], float]): Функция, которую нужно интерполировать.
        grid (list[float]): Список узлов интерполяции (значений аргумента функции).
        n (int, optional): Количество ближайших узлов для интерполяции.

    Returns:
        float: Значение многочлена в точке x.
    """
    func = np.vectorize(func)
    points = np.array(find_n_closest(x, grid, n))
    y_points = func(np.array(points))
    coefs = divided_differences(points, y_points)
    result = coefs[0]
    product = 1.0

    for i in range(1, n):
        product *= (x - points[i - 1])
        result += coefs[i] * product

    return result


def interpolation_error(x: float, grid: list, n: int = 2) -> list[list[int | float | Any] | list[float | Any]]:
    """Вычисляет оценку погрешности интерполяции для заданной точки x.

        Args:
            x (float): Точка, в которой оценивается погрешность интерполяции.
            grid (List[float]): Список узлов интерполяции (значений аргумента функции).
            n (int, optional): Количество ближайших узлов для оценки погрешности.

        Returns:
            List[List[Union[float, List[float]]]]: Список, содержащий:
                - [min_derivative, max_derivative]: Минимальное и максимальное значения
                  производной на интервале.
                - [Rmn, Rmx]: Оценки минимальной и максимальной погрешности интерполяции.
    """
    global min_derivative, max_derivative
    interval = find_n_closest(x, grid, 2)
    grid = find_n_closest(x, grid, n)
    if n == 2:
        derivative = lambda x: 2 + 1 / x ** 2
        min_derivative = derivative(interval[1])
        max_derivative = derivative(interval[0])

    elif n == 3:
        derivative = lambda x: -2 / x ** 3
        min_derivative = derivative(interval[1])
        max_derivative = derivative(interval[0])

    fact = factorial(n)
    prod = np.prod(list(map(lambda y: x - y, grid)))
    Rmn = min_derivative * prod / fact
    Rmx = max_derivative * prod / fact
    return [[min_derivative, max_derivative], [Rmn, Rmx]]


def finite_differences(x_values: list[float], y_values: list[float]) -> list[list[float]]:
    """
    Вычисляет конечные разности .

    Args:
        x_values (list): Список узлов интерполяции.
        y_values (list): Список значений функции в узлах.

    Returns:
        list[list[float]]: Матрица конечных разностей.
    """
    n = len(x_values)
    table = list(np.zeros((n, n)))
    for i in range(n):
        table[0][i] = y_values[i]

    for i in range(1, n):
        for j in range(n - i):
            table[i][j] = table[i - 1][j + 1] - table[i - 1][j]

    return table


def interpolation_newton_forward(x_values: list[float], y_values: list[float], x_interp: float) -> float:
    """
    Интерполяция по 1-й формуле Ньютона (конечные разности)
    Используется, когда x_interp немного находится близко к началу сетки.

    Args:
        x_values (list[float]): Список узлов (равномерно распределены)
        y_values (list[float]): Список значений функции в этих узлах
        x_interp (float): Точка, в которой нужно интерполировать значение
    Returns:
        float: Значение интерполированной функции
    """
    closest = find_closest_index(x_values, x_interp)
    if x_values[closest] <= x_interp:
        x_values = x_values[closest:]
        y_values = y_values[closest:]
    else:
        x_values = x_values[closest - 1:]
        y_values = y_values[closest - 1:]
    n = len(x_values)
    table = finite_differences(x_values, y_values)
    h = x_values[1] - x_values[0]
    t = (x_interp - x_values[0]) / h

    result = y_values[0]
    term = 1
    for i in range(1, n):
        term *= (t - (i - 1)) / i
        result += term * table[i][0]

    return result


def interpolation_newton_backward(x_values: list[float], y_values: list[float], x_interp: float) -> float:
    """
    Интерполяция по 2-й формуле Ньютона (конечные разности)
    Используется, когда x_interp немного находится близко к концу сетки.

    Args:
        x_values (list[float]): Список узлов (равномерно распределены)
        y_values (list[float]): Список значений функции в этих узлах
        x_interp (float): Точка, в которой нужно интерполировать значение
    Returns:
        float: Значение интерполированной функции
    """

    closest = find_closest_index(x_values, x_interp)
    if x_values[closest] >= x_interp:
        x_values = x_values[:closest + 1]
        y_values = y_values[:closest + 1]
    else:
        x_values = x_values[:closest + 2]
        y_values = y_values[:closest + 2]

    n = len(x_values)

    table = finite_differences(x_values, y_values)
    h = x_values[1] - x_values[0]
    t = (x_interp - x_values[-1]) / h

    result = y_values[-1]
    term = 1
    for i in range(1, n):
        term *= (t + (i - 1)) / i
        result += term * table[i][n - i - 1]

    return result


def gauss_forward(x_values: list[float], y_values: list[float], x_interp: float) -> float:
    """
    Первая формула Гаусса (прямые центральные разности)
    Используется, когда x_interp немного правее центрального узла.

    Args:
        x_values (list[float]): Список узлов (равномерно распределены)
        y_values (list[float]): Список значений функции в этих узлах
        x_interp (float): Точка, в которой нужно интерполировать значение
    Returns:
        float: Значение интерполированной функции
    """

    closest = find_closest_index(x_values, x_interp)
    if x_values[closest] > x_interp:
        closest -= 1
    n = len(x_values)

    h = x_values[1] - x_values[0]
    t = (x_interp - x_values[closest]) / h

    table = finite_differences(x_values, y_values)

    result = y_values[closest]
    term = 1
    for i in range(1, n):
        j = closest - i // 2
        if j == -1 or j >= n - i:
            break

        s = -1 if i % 2 == 0 else 1
        term *= (t + (i // 2) * s) / i
        result += term * table[i][j]

    return result


def gauss_backward(x_values: list[float], y_values: list[float], x_interp: float) -> float:
    """
    Вторая формула Гаусса (обратные центральные разности)
    Используется, когда x_interp немного левее центрального узла.

    Args:
        x_values (list[float]): Список узлов (равномерно распределены)
        y_values (list[float]): Список значений функции в этих узлах
        x_interp (float): Точка, в которой нужно интерполировать значение
    Returns:
        float: Значение интерполированной функции
    """

    closest = find_closest_index(x_values, x_interp)
    if x_values[closest] < x_interp:
        closest += 1

    n = len(x_values)

    h = x_values[1] - x_values[0]
    t = (x_interp - x_values[closest]) / h

    table = finite_differences(x_values, y_values)

    result = y_values[closest]
    term = 1
    for i in range(1, n):
        j = closest - (i + 1) // 2
        if j == -1 or j >= n - i:
            break

        s = 1 if i % 2 == 0 else -1
        term *= (t + (i // 2) * s) / i
        result += term * table[i][j]
    return result


def stirling(x_values: list[float], y_values: list[float], x_interp: float) -> float:
    """
        Формула Стирлинга
        Используется, когда x_interp находится очень близко к центральному узлу.

        Args:
            x_values (list[float]): Список узлов (равномерно распределены)
            y_values (list[float]): Список значений функции в этих узлах
            x_interp (float): Точка, в которой нужно интерполировать значение
        Returns:
            float: Значение интерполированной функции
        """
    closest = find_closest_index(x_values, x_interp)
    n = len(x_values)

    h = x_values[1] - x_values[0]
    t = (x_interp - x_values[closest]) / h

    table = finite_differences(x_values, y_values)

    result = y_values[closest] + t * (table[1][closest] + table[1][closest - 1]) / 2
    term1 = 1
    term2 = t
    fctrl = 1
    for i in range(2, n):
        fctrl *= i
        if i % 2 == 0:
            j = closest - i // 2
            if j == -1 or j >= n - i:
                break
            term1 *= (t ** 2 - ((i - 1) // 2) ** 2)
            result += term1 * table[i][j] / fctrl
        else:
            j1 = closest - i // 2
            j2 = closest - (i + 1) // 2
            if j1 == -1 or j1 >= n - i or j2 == -1 or j2 >= n - i:
                break
            term2 *= (t ** 2 - ((i - 1) // 2) ** 2)
            result += term2 * (table[i][j1] + table[i][j2]) / 2 / fctrl

    return result


def bessel(x_values: list[float], y_values: list[float], x_interp: float) -> float:
    """
        Формула Бесселя
        Используется, когда x_interp находится очень равноудалено от двух центральных узлов.

        Args:
            x_values (list[float]): Список узлов (равномерно распределены)
            y_values (list[float]): Список значений функции в этих узлах
            x_interp (float): Точка, в которой нужно интерполировать значение
        Returns:
            float: Значение интерполированной функции
        """
    closest = find_closest_index(x_values, x_interp)
    n = len(x_values)

    h = x_values[1] - x_values[0]
    t = (x_interp - x_values[closest]) / h

    if x_interp < x_values[closest]:
        closest -= 1

    table = finite_differences(x_values, y_values)

    result = (y_values[closest] + y_values[closest + 1]) / 2

    term1 = 1
    fctrl = 1

    for i in range(1, n):
        fctrl *= i
        if i % 2 == 0:
            j1 = closest - i // 2
            j2 = closest - i // 2 + 1
            if j1 == -1 or j1 >= n - i or j2 == -1 or j2 >= n - i:
                break
            term1 *= (t + (i - 1) // 2) * (t - i // 2)
            result += term1 * (table[i][j1] + table[i][j2]) / 2 / fctrl
        else:
            j = closest - i // 2
            if j == -1 or j >= n - i:
                break
            term2 = term1 * (t - 0.5)
            result += term2 * table[i][j] / fctrl

    return result


func = lambda x: x ** 3 - cos(2 * x)

interval = [0.1, 0.6]

n = 11

grid = np.linspace(interval[0], interval[1], n)
print(grid)

x_values = list(grid)
y_values = list(map(func, x_values))

values = [0.12, 0.58, 0.33]

# Вывод таблицы конечных разностей
# table = finite_differences(x_values, y_values)
# print("\nТаблица конечных разностей:")
# separator = "-" * (12 * (n + 1) + n)
# header = f"{'y':^{12}}|"
# for i in range(1, n):
#     header += f"{f'Δ^{i}y':^{12}}|"
# print(separator)
# print(header)
# print(separator)
# for j in range(n):
#     row = f"{table[0][j]:^{12}.4f}|"
#     for i in range(1, n - j):
#         row += f"{table[i][j]:^{12}.4f}|"
#     print(row)
# print(separator)
#
# print(func(values[0]), interpolation_newton_forward(x_values, y_values, values[0]),
#       func(values[0]) - interpolation_newton_forward(x_values, y_values, values[0]))
#
# print(func(values[1]), interpolation_newton_backward(x_values, y_values, values[1]),
#       func(values[1]) - interpolation_newton_backward(x_values, y_values, values[1]))
#
# print(func(values[2]), gauss_backward(x_values, y_values, values[2]),
#       func(values[2]) - gauss_backward(x_values, y_values, values[2]))
#
#
# min_der = 0
# max_der = -2048
# fctrl = factorial(11)
# w1 = 1
# w2 = 1
# w3 = 1
# for i in range(n):
#     w1 *= values[0] - x_values[i]
#     w2 *= values[1] - x_values[i]
#     w3 *= values[2] - x_values[i]
#
# min_R = 0
# max_R1 = max_der*w1/fctrl
# max_R2 = max_der*w2/fctrl
# max_R3 = max_der*w3/fctrl
#
# print(max_R1, max_R2, max_R3, sep="\n")