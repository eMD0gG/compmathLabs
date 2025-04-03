from math import *
import numpy as np
from typing import Callable, Optional


def find_n_closest(target: float,
                   grid: list[float],
                   n: int) -> list[float]:
    """
    Ищет n ближайших к target значений из сетки grid.
    Результат отсортирован по возрастанию.

    Args:
        target (float): Число, к которому ищутся ближайшие значения.
        grid (list[float]): Список чисел для поиска.
        n (int): Количество ближайших значений.

    Returns:
        List[float]: Отсортированный список из n ближайших чисел.
    """
    sorted_grid = sorted(grid, key=lambda x: abs(x - target))[:n]
    sorted_grid.sort()
    return sorted_grid


def divided_differences(x_points: float, y_points: float) -> list[float]:
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
                grid: list[float],
                n: int = 11) -> float | None:
    """Вычисляет интерполяционного многочлена Лагранжа в точке x.

       Args:
           x (float): Точка, в которой вычисляется интерполированное значение.
           func (Callable[[float], float]): Функция, которую нужно интерполировать.
           grid (list[float]): Список узлов интерполяции (значений аргумента функции).
           n (int, optional): Количество ближайших узлов для интерполяции.

       Returns:
           Optional[float]: Интерполированное значение или None.
     """
    points = np.array(find_n_closest(x, grid, n))
    result = 0
    for i in range(n):
        j = np.arange(n) != i
        p = np.prod((x - points[j]) / (points[i] - points[j]))
        result += func(points[i]) * p

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


def interpolation_error(x: float, grid: list, n: int = 2) -> list[list[float], list[float]]:
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


func = lambda x: x**2 - cos(0.5 * np.pi * x)

interval = [0.4, 0.9]
grid = np.linspace(interval[0], interval[1], 11)

values = [0.64]
n = 3
for j in values:
    print(lagrange_interpolate(j,func,grid,n), func(j), func(j) - lagrange_interpolate(j, func, grid, n),)
print('\n')

for j in values:
    print(newton_interpolate(j, func, grid, n), func(j), func(j) - newton_interpolate(j, func, grid, n))
