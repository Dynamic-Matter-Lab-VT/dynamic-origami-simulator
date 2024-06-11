import numpy as np
from scipy.integrate import solve_ivp
import pickle
import matplotlib.pyplot as plt

n = 10
ks = np.ones(n) * 2
ms = np.ones(n) * 1
bs = np.ones(n) * 0.005
x0 = np.zeros(n)


def f(t, y):
    x = y[:n]
    v = y[n:]
    a = np.zeros(n)

    # |--(k0,b0)--m1--(k1,b1)--m2--(k2,b2)--m3- ... -mn ->

    # spring force + damping force + excitation force

    def f_s(k, dx):
        return k * dx + 1.0 * dx ** 3
        # return k * np.sin(dx) / 50

    def f_d(b, dv):
        return b * dv
        # return 0

    def f_e(t_):
        # if t_ < 10:
        #     # return t_
        #     return 10.0
        # else:
        #     return 0
        # return 10.0 * np.sin(10 * np.pi * t_)
        return 100.0

    if n == 1:
        a[0] = (-f_s(ks[0], x[0])  # spring force
                - f_d(bs[0], v[0])  # damping force
                + f_e(t)) / ms[0]  # excitation force
    else:
        a[0] = (-f_s(ks[0], x[0]) + f_s(ks[1], x[1] - x[0])  # spring force
                - f_d(bs[0], v[0]) + f_d(bs[1], v[1] - v[0])  # damping force
                + f_e(t)) / ms[0]  # excitation force

        # v[0] = 2.0 * np.sin(10 * np.pi * t)

        for i in range(1, n - 1):
            a[i] = (-f_s(ks[i - 1], x[i] - x[i - 1]) + f_s(ks[i], x[i + 1] - x[i])  # spring force
                    - f_d(bs[i - 1], v[i] - v[i - 1]) + f_d(bs[i], v[i + 1] - v[i])  # damping force
                    + 0) / ms[i]

        a[-1] = (-f_s(ks[-2], x[-1] - x[-2])  # spring force
                 - f_d(bs[-2], v[-1] - v[-2])  # damping force
                 + 0) / ms[-1]  # excitation force

    return np.concatenate([v, a])


def get_fd_curve(sol):
    m = 0
    ys = sol.y
    ts = sol.t

    plt.figure()
    for i in range(len(ts)):
        a = f(ts[i], ys[:, i])
        # plt.plot(ys[m, i], a[m + n], 'ko')
        plt.plot(ys[m, i], 10.0 * np.sin(10 * np.pi * ts[i]), 'ko')

    plt.xlim(-1, 1)

    plt.xlabel('Displacement')
    plt.ylabel('Acceleration')
    plt.title('Force Displacement Curve')
    plt.show()


def animate(sol):
    t = sol.t
    x = sol.y[:n].T
    x0 = np.linspace(0, 10, n)
    # draw masses as circles at x0 + x and animate as time progresses
    # draw springs as lines

    print(x.shape)

    for i in range(len(t)):
        plt.plot(x0 + x[i] * 10, np.zeros(n), 'ko')
        plt.plot(x0 + x[i] * 10, np.zeros(n), 'k-')
        plt.ion()
        plt.show()
        # set x limits to 0, 10
        plt.xlim(-2, 12)
        plt.pause(0.001)
        plt.clf()


def fourier_analysis(sol):
    t = sol.t
    x = sol.y[:n].T
    # find fft of x and plot
    X = np.fft.fft(x, axis=0)
    X = np.abs(X)
    X = X[0:int(X.shape[0] / 2)]
    plt.figure()
    plt.plot(X[2:, :])
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Analysis')
    plt.show()


if __name__ == '__main__':
    t = np.linspace(0, 60, 1000)
    y0 = np.concatenate([x0, np.zeros(n)])
    solution = solve_ivp(f, (0, t[-1]), y0, t_eval=t)

    x = solution.y[:n].T
    plt.plot(solution.t, x)
    plt.show()

    # animate(solution)
    # get_fd_curve(solution)
    fourier_analysis(solution)
