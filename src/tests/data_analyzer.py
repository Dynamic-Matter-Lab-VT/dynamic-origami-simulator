import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def get_corelation_estimates(res_freq, freq):
    # x and y are 1D arrays
    s = np.linspace(0, 0.061340, 100)

    x = np.sin(2 * np.pi * freq * s) * np.sin(2 * np.pi * res_freq * s)

    # plot x vs s
    plt.figure()
    plt.plot(s, x)
    plt.xlabel('s')
    plt.ylabel('x')
    plt.title('x vs s')
    plt.show()

    # get x*xtranspose matrix
    mat = np.outer(x, x)
    # show image of x*xtranspose matrix
    plt.figure()
    plt.imshow(mat, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('x')
    plt.title('x*xtranspose')
    plt.show()


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'SimpleSpring.pkl'
    filename = data_dir + 'TaperedSpring_sq_nl20.0_hz.pkl'
    # freq = str(input())
    # freq = '20_'
    # filename = data_dir + 'TaperedSpring_sq' + freq + 'hz.pkl'
    #
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
    # plt.imshow(X, aspect='auto', cmap='jet')
    # contour plot
    plt.contourf(X, cmap='flag', levels=100)
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

    # get_corelation_estimates(40, 8.734)