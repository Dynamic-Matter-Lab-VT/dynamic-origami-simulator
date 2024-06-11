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


def frequency_data():
    # generate sinusiodal data for incresing frequency from 1 to 20 Hz
    n = 30000
    t = np.linspace(0, 10, n)
    data = np.sin(8 * np.pi * t ** 2)
    # data = np.zeros(n)
    # for i in range(1, 101):
    #     data += np.sin(2 * np.pi * i * t)
    data = data / np.max(data)
    plt.figure('Data')
    plt.plot(t, data)
    plt.show()

    # plot frequency spectrum
    freq = np.fft.fftfreq(n, t[1] - t[0])
    freq = freq[:n // 2]
    data_fft = np.fft.fft(data)
    data_fft = data_fft[:n // 2]
    plt.figure('Frequency Spectrum')
    plt.plot(freq, np.abs(data_fft))
    plt.show()

    # get velocity data
    data_grad = np.gradient(data, np.linspace(0, 1, n))
    data_dot = data_grad / np.max(data_grad)

    pred = cumtrapz(data_dot, np.linspace(0, 1, n), initial=0)
    pred = pred / np.max(pred)

    plt.figure('Predicted Data')
    plt.plot(t, data)
    plt.plot(t, data_dot)
    plt.plot(t, pred)
    plt.legend(['Data', 'Data_dot', 'Predicted'])
    plt.show()

    # get frequency vs time as in how frequency changes with time
    freq = np.zeros(n)
    # get freqency of 10 points at a time and store it in freq
    for i in range(n - 5):
        chunk = data[i:i + 5]
        # get fft and frequency of the chunk and store it in freq
        frequencies = np.fft.fftfreq(5, t[1] - t[0])
        frequencies = frequencies[:5 // 2]
        chunk_fft = np.fft.fft(chunk)
        chunk_fft = chunk_fft[:5 // 2]
        freq[i] = frequencies[np.argmax(np.abs(chunk_fft))]

    plt.figure('Frequency vs Time')
    plt.plot(t, freq)
    plt.show()

    # # save the data and its derivative as terrain data and velocity data
    # np.save('terrain_data_freq.npy', data)
    # np.save('velocity_data_freq.npy', data_dot)


def load_data():
    global terrain_data, velocity_data, n
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/tapered_spring_tests/'
    # terrain_data = np.load(data_dir + 'terrain_data.npy')
    # velocity_data = np.load(data_dir + 'velocity_data.npy')
    terrain_data = np.load(data_dir + 'terrain_data_freq.npy')
    velocity_data = np.load(data_dir + 'velocity_data_freq.npy')
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
    # generate_random_terrain()
    # load_data()
    # t = np.linspace(0, 10, 1000)
    # plt.plot(t, get_velocity(t))
    # plt.plot(t, get_noisy_velocity(t))
    # plt.show()
    frequency_data()
