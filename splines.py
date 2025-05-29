import numpy as np
import matplotlib.pyplot as plt
from thomas_algorithm import thomas_algorithm

class CubicSpline:
    def __init__(self, x, y, y0_prime, yN_prime):
        """Инициализация кубического сплайна с заданными краевыми производными.

            Args:
                x (np.ndarray): абсциссы узлов
                y (np.ndarray): ординаты узлов
                y0_prime (np.ndarray): производная в первом узле
                yN_prime (np.ndarray): производная в последнем узле
        """
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n = len(x) - 1
        self.h = np.diff(self.x)

        self.y0_prime = float(y0_prime)
        self.yN_prime = float(yN_prime)


        self.A = np.zeros(self.n + 1)
        self.B = np.zeros(self.n + 1)
        self.C = np.zeros(self.n + 1)
        self.F = np.zeros(self.n + 1)


        for i in range(1, self.n):
            self.A[i] = self.h[i-1] / (self.h[i-1] + self.h[i])
            self.B[i] = 2.0
            self.C[i] = self.h[i] / (self.h[i-1] + self.h[i])
            self.F[i] = 3 * (self.C[i] * (self.y[i+1] - self.y[i]) / self.h[i] +
                             self.A[i] * (self.y[i] - self.y[i-1]) / self.h[i-1])

        self.B[0] = 1.0
        self.C[0] = 0.0
        self.F[0] = self.y0_prime

        self.B[self.n] = 1.0
        self.A[self.n] = 0.0
        self.F[self.n] = self.yN_prime

        self.m = thomas_algorithm(self.A, self.B, self.C, self.F)

        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.d = np.zeros(self.n)

        for i in range(self.n):
            self.a[i] = self.y[i]
            self.b[i] = self.m[i]
            self.c[i] = (3 * (self.y[i+1] - self.y[i]) / self.h[i] - 2 * self.m[i] - self.m[i+1]) / self.h[i]
            self.d[i] = (2 * (self.y[i] - self.y[i+1]) / self.h[i] + self.m[i] + self.m[i+1]) / (self.h[i] ** 2)


    def __call__(self, x_eval):
        """Функция подсчета значения в точках с ипользованием полученных ранее коэффициентов сплайнов
               Args:
                   x_val (np.ndarray): массив точек (можно передать просто скаляр)
               Returns:
                   np.ndarray: массив значений (если входные данные скаляр, то на выходе будет скаляр)
               """

        x_eval = np.asarray(x_eval, dtype=float)

        indices = np.searchsorted(self.x, x_eval) - 1
        indices = np.clip(indices, 0, self.n - 1)

        dx = x_eval - self.x[indices]
        a = self.a[indices]
        b = self.b[indices]
        c = self.c[indices]
        d = self.d[indices]

        return a + b * dx + c * dx**2 + d * dx**3

x = [-2, -1, 0, 1, 2]
y = [12, 1, 0, 3, 28]

func = lambda x: x**4 + x**3 + x**2

spline = CubicSpline(x, y, -24, 48)

x_dense = np.linspace(-2, 2, 200)
y_dense = spline(x_dense)
y_true_dense = func(x_dense)

plt.figure(figsize=(8, 5))
plt.plot(x_dense, y_dense, label='Natural Cubic Spline', linewidth=2)
plt.plot(x_dense, y_true_dense, label=r'$x^4$', linestyle='--', linewidth=2, color='green')
plt.plot(x, y, 'o', label='Nodes', color='black')

plt.title('Сравнение кубического сплайна и функции $x^4 + x^3 + x^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()