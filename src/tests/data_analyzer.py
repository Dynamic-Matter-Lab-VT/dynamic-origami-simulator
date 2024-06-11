import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    # filename = data_dir + 'TaperedSpring_lin_terrain_class' + str(10.0) + '_hz.pkl'
    # filename = data_dir + 'TaperedSpring_nlin_terrain_class' + str(10.0) + '_hz.pkl'
    # filename = data_dir + 'TaperedSpring_linearity_test.pkl'
    # filename = data_dir + 'TaperedSpring_freq_analysis.pkl'
    # filename = data_dir + 'CylinderSpring_impulse_response.pkl'
    filename = data_dir + 'CylinderSpring_cubic_nonlinear.pkl'
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
        x[:, :, i] = solution.y[:i_max * 3, i].reshape((i_max, 3)) - solution.y[0:i_max * 3, 2].reshape((i_max, 3))

    x[:, 2, :] = x[:, 2, :] - x[0, 2, :]
    test_n = 99

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

    # plot same fft in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ = np.arange(X.shape[1])
    y_ = np.arange(X.shape[0])
    X_, Y_ = np.meshgrid(x_, y_)
    ax.plot_surface(X_, Y_, X, cmap='viridis')
    plt.show()

    # plt.figure()
    # plt.imshow(x[:, 2, :], aspect='auto', cmap='jet')
    # plt.colorbar()
    # plt.xlabel('time')
    # plt.ylabel('displacement')
    # plt.title('Displacement vs Time')
    # plt.show()

    # show displacement of last node
    plt.figure()
    print(x.shape)
    for i in range(x.shape[0]):
        plt.plot(t, x[i, 2, :])
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Displacement vs Time')
    plt.show()

    # get_corelation_estimates(40, 8.734)
