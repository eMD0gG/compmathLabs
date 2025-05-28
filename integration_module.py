from math import cos, sin

import numpy as np


def rectangle_integration(interval, n, func, epsilon, type='left'):
    x_values = np.linspace(interval[0], interval[1], n)
    y_values = list(map(func, x_values))
    h = (x_values[-1] - x_values[0]) / n
    if type == 'left':
        I2 = h * sum(y_values[:-1])
    elif type == 'right':
        I2 = h * sum(y_values[1:])
    elif type == 'center':
        np.linspace(interval[0] + h / 2, interval[-1] - h / 2, n - 1)
        y_values = list(map(func, x_values))
        I2 = h * sum(y_values[1:])
    else:
        print('No such type of rectangle integration!')
        return None

    I1 = float('inf')

    while abs(I1 - I2) > epsilon:
        I1 = I2
        n *= 2
        x_values = np.linspace(interval[0], interval[1], n)
        y_values = list(map(func, x_values))
        h = (x_values[-1] - x_values[0]) / n
        if type == 'left':
            I2 = h * sum(y_values[:-1])
        elif type == 'right':
            I2 = h * sum(y_values[1:])
        elif type == 'center':
            np.linspace(interval[0] + h / 2, interval[-1] - h / 2, n - 1)
            y_values = list(map(func, x_values))
            I2 = h * sum(y_values[1:])

    return I2


def trap_integration(interval, n, func, epsilon):
    x_values = np.linspace(interval[0], interval[1], n)
    y_values = list(map(func, x_values))
    h = (x_values[-1] - x_values[0]) / n
    I2 = h * ((y_values[0] + y_values[-1]) / 2 + sum(y_values[1:-1]))

    I1 = float('inf')

    while abs(I1 - I2) > epsilon:
        I1 = I2
        n *= 2
        x_values = np.linspace(interval[0], interval[1], n)
        y_values = list(map(func, x_values))
        h = (x_values[-1] - x_values[0]) / n
        I2 = h * sum(y_values[1:])

    return I2


def simpson_integration(interval, n, func, epsilon):
    x_values = np.linspace(interval[0], interval[1], n)
    y_values = list(map(func, x_values))
    h = (x_values[-1] - x_values[0]) / n
    x_center = np.linspace(interval[0] + h / 2, interval[-1] - h / 2, n - 1)
    y_center = list(map(func, x_center))
    I2 = h * (y_values[0] + y_values[-1] + 2 * sum(y_values[1:-1]) + 4 * sum(y_center)) / 6

    I1 = float('inf')

    while abs(I1 - I2) > epsilon:
        I1 = I2
        n *= 2
        x_values = np.linspace(interval[0], interval[1], n)
        y_values = list(map(func, x_values))
        h = (x_values[-1] - x_values[0]) / n / 6
        x_center = np.linspace(interval[0] + h / 2, interval[-1] - h / 2, n - 1)
        y_center = list(map(func, x_center))
        I2 = h * (y_values[0] + y_values[-1] + 2 * sum(y_values[1:-1]) + 4 * sum(y_center[:-1]))

        h = (x_values[-1] - x_values[0]) / n
        I2 = h * sum(y_values[1:])

    return I2


def weddle_integration(interval, n, func, epsilon):
    x_values = np.linspace(interval[0], interval[1], n)
    y_values = list(map(func, x_values))
    h = (x_values[-1] - x_values[0]) / n
    I2 = 0.3 * h * (
            2 * sum(y_values[::6]) + 5 * sum(y_values[1::6]) + sum(y_values[2::6]) + 6 * sum(y_values[3::6]) + sum(
        y_values[4::6]) + 5 * sum(y_values[5::6]) - y_values[0] - y_values[-1])

    I1 = float('inf')

    while abs(I1 - I2) > epsilon:
        I1 = I2
        n *= 2
        x_values = np.linspace(interval[0], interval[1], n)
        y_values = list(map(func, x_values))
        h = (x_values[-1] - x_values[0]) / n
        I2 = 0.3 * h * (
                2 * sum(y_values[::6]) + 5 * sum(y_values[1::6]) + sum(y_values[2::6]) + 6 * sum(y_values[3::6]) + sum(
            y_values[4::6]) + 5 * sum(y_values[5::6]) - y_values[0] - y_values[-1])

    return I2


def newton_cotes_interpolation_value(interval, c, func):
    n = len(c)
    x_values = np.linspace(interval[0], interval[1], n)
    y_values = list(map(func, x_values))

    I = (interval[1] - interval[0]) * (sum(c[i] * y_values[i] for i in range(n)))
    return I


def newton_cotes_interpolation(interval, n, c, func, epsilon):
    I2 = newton_cotes_interpolation_value(interval, c, func)
    I1 = float('inf')

    count = 1

    while abs(I1 - I2) > epsilon:
        I1 = I2
        count *= 2
        interval = np.linspace(interval[0], interval[-1], count + 1)
        I2 = 0
        for i in range(count):
            I2 += newton_cotes_interpolation_value([interval[i], interval[i + 1]], c, func)

    return I2


func = lambda x: x ** 3 - np.cos(2 * x)
integral = lambda x: (x ** 4) / 4 - np.sin(2 * x) / 2

interval = [0.1, 0.6]

n = 11

# Вычисление интегралов разными методами
I_exact = integral(interval[1]) - integral(interval[0])
Isimp = simpson_integration(interval, 4, func, 0.0000001)
Itrap = trap_integration(interval, 4, func, 0.0000001)
Irect = rectangle_integration(interval, 4, func, 0.0000001)

# Репрезентативынй результат
print("\n" + "="*60)
print(f"{'Метод':<15} | {'Значение':^15} | {'Погрешность':^15}")
print("-"*60)
print(f"{'Точное':<15} | {I_exact:^15.6f} | {'—':^15}")
print(f"{'Симпсона':<15} | {Isimp:^15.6f} | {abs(I_exact - Isimp):^15.6f}")
print(f"{'Трапеций':<15} | {Itrap:^15.6f} | {abs(I_exact - Itrap):^15.6f}")
print(f"{'Прямоугольников':<15} | {Irect:^15.6f} | {abs(I_exact - Irect):^15.6f}")
print("="*60 + "\n")