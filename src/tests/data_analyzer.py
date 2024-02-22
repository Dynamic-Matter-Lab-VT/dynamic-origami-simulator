import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'SimpleSpring.pkl'
    # filename = data_dir + 'TaperedSpring_0.pkl'
    # freq = str(input())
    freq = '20_'
    filename = data_dir + 'TaperedSpring_sq' + freq + 'hz.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    i_max = 100
    t = solution.t

    x = np.zeros((i_max, 3, t.shape[0]))
    for i in range(0, t.shape[0]):
        x[:, :, i] = solution.y[:i_max * 3, i].reshape((i_max, 3)) - solution.y[0:i_max * 3, 0].reshape((i_max, 3))

    test_n = 0

    plt.figure()
    plt.plot(t, x[test_n, 2, :])
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Displacement vs Time')
    plt.show()

    # stack all ffts of displacement in matrix and show image
    X = np.fft.fft(x[:, 2, :], axis=1)
    X = np.abs(X)
    X = X[:, 0:int(X.shape[1] / 2)]

    plt.figure()
    plt.imshow(X, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('Frequency')
    plt.ylabel('Node')
    plt.title('FFT of Displacement')
    plt.show()


    plt.figure()
    plt.imshow(x[:, 2, :], aspect='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.title('Displacement vs Time')
    plt.show()
