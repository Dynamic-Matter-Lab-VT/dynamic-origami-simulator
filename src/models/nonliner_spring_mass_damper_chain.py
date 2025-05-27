import numpy as np
from scipy.integrate import solve_ivp
import pickle
import matplotlib.pyplot as plt
from benchmark_utils import input_func
import os
from tqdm import tqdm

data_dir = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))) + '/../data/nl_spring_mass_damper_raw_data/'


def experiment(n):
    ks = np.ones(n) * 1.0
    ms = np.ones(n) * 1
    bs = np.ones(n) * 0.0
    x0 = np.linspace(0, n - 1, n)
    dt = 0.01

    # @jit(nopython=True)
    def f(t, y):
        x = y[:n]
        v = y[n:]
        a = np.zeros(n)

        # |--(k0,b0)--m1--(k1,b1)--m2--(k2,b2)--m3- ... -mn ->

        # spring force + damping force + excitation force

        def f_s(k, dx):
            return k * dx
            # return k * dx + 10000 * k * dx ** 3 + 100000 * k * dx ** 5
            # return k * ((dx - 1) / 1.0) ** 5
            # return k * np.sin(dx) / 50

        def f_d(b, dv):
            return b * dv
            # return 0

        def f_e(t_):
            # if t_ < 1:
            #     # return t_
            #     return 0.0
            # else:
            #     return 0.1
            # return 10.0 * np.sin(10 * np.pi * t_)
            # return 100.0
            return input_func(t_)

        v[0] = (-f_e(t - 2 * dt) + 8 * f_e(t - dt) - 8 * f_e(t + dt) + f_e(t + 2 * dt)) / (12 * dt)
        if n == 1:

            a[0] = (-f_s(ks[0], x[0])  # spring force
                    - f_d(bs[0], v[0])  # damping force
                    # + f_e(t)) / ms[0]  # excitation force
                    + 0) / ms[0]
        else:
            a[0] = (-f_s(ks[0], x[0]) + f_s(ks[1], x[1] - x[0])  # spring force
                    - f_d(bs[0], v[0]) + f_d(bs[1], v[1] - v[0])  # damping force
                    # + f_e(t)) / ms[0]  # excitation force
                    + 0) / ms[0]

            # v[0] = 2.0 * np.sin(10 * np.pi * t)

            for i in range(1, n - 1):
                a[i] = (-f_s(ks[i - 1], x[i] - x[i - 1]) + f_s(ks[i], x[i + 1] - x[i])  # spring force
                        - f_d(bs[i - 1], v[i] - v[i - 1]) + f_d(bs[i], v[i + 1] - v[i])  # damping force
                        + 0) / ms[i]

            a[-1] = (-f_s(ks[-2], x[-1] - x[-2])  # spring force
                     - f_d(bs[-2], v[-1] - v[-2])  # damping force
                     + 0) / ms[-1]  # excitation force

        return np.concatenate([v, a])

    def save_solution(sol):
        sol_y = sol.y[:n]
        # subtract the initial condition
        sol_y = sol_y - sol_y[:, 0].reshape(-1, 1)
        # sol_y = sol_y[1:]

        data = {'t': sol.t,
                'y': sol_y,
                'n': n,
                'ks': ks,
                'ms': ms,
                'bs': bs,
                'ip': input_func(sol.t)}
        # filename = data_dir + 'quintic_%d.pkl' % n
        filename = data_dir + 'pure_sine_%d.pkl' % n
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    t = np.linspace(0, 100, 10000)
    y0 = np.concatenate([x0, np.zeros(n)])
    solution = solve_ivp(f, (0, t[-1]), y0, t_eval=t)
    save_solution(solution)


def fourier_analysis(t, y, n):
    x = y[:n].T
    X = np.fft.fft(x, axis=0)
    X = np.abs(X)
    X = X[0:int(X.shape[0] / 2)]
    plt.figure()
    plt.plot(X[2:, :])
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Analysis')
    plt.show()


def animate(t, y, n):
    x = y[:n].T
    x0 = np.linspace(0, 1, n + 1)

    plt.figure()

    for i in range(len(t)):
        plt.plot(0, 0, 'r*')
        plt.plot(np.zeros(2), np.array([0, x0[1] + x[i, 0]]), 'b-')
        plt.plot(np.zeros(n), x0[1:n + 1] + x[i], 'ko')
        plt.plot(np.zeros(n), x0[1:n + 1] + x[i], 'b-')
        plt.text(0.8, 1.0, 't: %.1f' % t[i], fontsize=12)
        plt.ion()
        plt.show()
        plt.ylim(-0.01, 1.2)
        plt.xlim(-1, 1)
        plt.pause(0.001)
        plt.clf()

    plt.show()
    plt.ioff()


def plot(t, y, n, filename='test'):
    x = y[:n].T
    plt.figure(filename)
    for i in range(n):
        plt.plot(t, x[:, i], c=plt.cm.jet(i / n))
    # plt.show()


if __name__ == '__main__':
    experiment(5)
    # num_exp = 100
    # bar = tqdm(total=num_exp)
    # for n in range(1, num_exp + 1):
    #     experiment(n)
    #     bar.update(1)

    # read solution example
    with open(data_dir + 'quintic_100.pkl', 'rb') as f:
        data = pickle.load(f)

    t = data['t']
    y = data['y']
    n = data['n']
    ip = data['ip']

    # plt.plot(t, ip)
    plot(t, y, n, 'quintic')
    # animate(t, y, n)
    # fourier_analysis(t, y, n)

    plt.show()
