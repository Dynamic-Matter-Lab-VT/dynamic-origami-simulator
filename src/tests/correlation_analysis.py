import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    for freq in [10, 20, 30, 40, 50]:
        filename = data_dir + 'TaperedSpring_' + str(freq) + 'hz.pkl'

        with open(filename, 'rb') as f:
            solution = pickle.load(f)

        i_max = 100
        t = solution.t

        x = np.zeros((i_max, 3, t.shape[0]))
        data = np.zeros((i_max, t.shape[0]))
        for i in range(t.shape[0]):
            x[:, :, i] = solution.y[:i_max * 3, i].reshape((i_max, 3)) - solution.y[0:i_max * 3, 0].reshape((i_max, 3))
            data[:, i] = x[:, 2, i]

        for i in range(i_max):
            data[i, :] = data[i, :] / np.linalg.norm(data[i, :])

        # corr = np.cov(data)
        corr = np.matmul(data, data.T)

        fig = plt.figure('Correlation Matrix ' + str(freq) + ' Hz')
        plt.imshow(corr, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.show()
        # save in high resolution
        plot_name = data_dir + '../plots/correlation_matrix_' + str(freq) + 'hz.png'
        fig.savefig(plot_name, dpi=300)

        scores = np.zeros((i_max, 1))
        for i in range(i_max):
            scores[i] = np.sum(corr[i, :]) - corr[i, i]

        fig = plt.figure('Scores ' + str(freq) + ' Hz')
        plt.plot(scores)
        plt.xlabel('Sensor Number')
        plt.ylabel('Score')
        plt.title('Scores ' + str(freq) + ' Hz')
        plt.show()

        plot_name = data_dir + '../plots/scores_' + str(freq) + 'hz.png'
        fig.savefig(plot_name, dpi=300)
