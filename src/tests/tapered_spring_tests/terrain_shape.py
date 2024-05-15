import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import os

terrain_data = None
velocity_data = None
n = None


def generate_random_terrain():
    # smooth random 1d timeseries data
    n = 30000
    m = 5
    data = 1 - 2 * np.random.rand(n)
    for i in range(1000):
        data = np.convolve(data, np.ones(m), mode='same') / m

    data = data / np.max(data)
    data_grad = np.gradient(data, np.linspace(0, 1, n))
    data_dot = data_grad / np.max(data_grad)

    pred = cumtrapz(data_dot, np.linspace(0, 1, n), initial=0)
    pred = pred / np.max(pred)
    plt.plot(data)
    plt.plot(data_dot)
    plt.plot(pred)
    plt.legend(['data', 'data_dot', 'pred'])
    plt.show()

    # save the data and its derivative as terrain data and velocity data
    np.save('terrain_data.npy', data)
    np.save('velocity_data.npy', data_dot)


def load_data():
    global terrain_data, velocity_data, n
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/tapered_spring_tests/'
    terrain_data = np.load(data_dir + 'terrain_data.npy')
    velocity_data = np.load(data_dir + 'velocity_data.npy')
    n = len(terrain_data)


def get_velocity(t_):
    global velocity_data
    idxs = np.floor(t_ * 1000).astype(int)
    return velocity_data[idxs]


def get_noisy_velocity(t_):
    global velocity_data
    idxs = np.floor(t_ * 1000).astype(int)
    # add some white noise to the velocity data
    return velocity_data[idxs] + 0.1 * np.random.random()


def get_terrian(t_):
    global terrain_data
    idxs = np.floor(t_ * 1000).astype(int)
    return terrain_data[idxs]
    # return 0


if __name__ == "__main__":
    generate_random_terrain()
    load_data()
    t = np.linspace(0, 10, 1000)
    plt.plot(t, get_velocity(t))
    plt.plot(t, get_noisy_velocity(t))
    plt.show()
