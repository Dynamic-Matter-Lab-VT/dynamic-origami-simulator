import numpy as np
import matplotlib.pyplot as plt
from src.tests.tapered_spring_tests.terrain_shape import *
import pickle
import tensorflow as tf


def create_readout_layer(input_shape, weights=None, biases=None):
    m = tf.keras.models.Sequential()
    if weights is not None and biases is not None:
        m.add(tf.keras.layers.Dense(1, input_shape=input_shape, use_bias=True, weights=[weights, biases]))
    else:
        m.add(tf.keras.layers.Dense(1, input_shape=input_shape, use_bias=True))
    return m


if __name__ == "__main__":
    tf.autograph.set_verbosity(3)
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../../data/simulations/'
    filename = data_dir + 'TaperedSpring_lin_terrain_class' + str(10.0) + '_hz.pkl'
    # filename = data_dir + 'TaperedSpring_nlin_terrain_class' + str(10.0) + '_hz.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    i_max = 100
    t = solution.t

    x = np.zeros((i_max, 3, t.shape[0]))
    for i in range(0, t.shape[0]):
        x[:, :, i] = solution.y[:i_max * 3, i].reshape((i_max, 3)) - solution.y[0:i_max * 3, 2].reshape((i_max, 3))

    x[:, 2, :] = x[:, 2, :] - x[0, 2, :]

    # # show each readout point one by one
    # for i in range(x.shape[0]):
    #     plt.figure()
    #     plt.plot(t, x[i, 2, :])
    #     plt.xlabel('Time')
    #     plt.ylabel('Displacement')
    #     plt.title('Displacement vs Time')
    #     plt.show()

    readout_n = 10
    # choose equidistant points from x spaced by readout_n
    readout = x[::readout_n, :, :] * 10
    weights = np.ones(readout.shape[0])
    biases = np.zeros(readout.shape[0])

    load_data()
    data = get_terrian(t)

    # plot the terrain data
    plt.figure()
    plt.plot(t, data)
    plt.xlabel('Time')
    plt.ylabel('Terrain')
    plt.title('Terrain vs Time')
    plt.show()

    # plot the displacement of the readout points
    plt.figure()
    for i in range(readout.shape[0]):
        plt.plot(t, readout[i, 2, :])
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Displacement vs Time')
    plt.show()

    model = create_readout_layer((readout.shape[0],))
    model.summary()

    x_train = readout[:, 2, :].T
    y_train = data

    print(x_train.shape, y_train.shape)
    model.compile(optimizer='sgd', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=10000, verbose=1, batch_size=1000)

    # save the weights and biases
    learned_weights, learned_biases = model.layers[0].get_weights()
    # save
    np.save('learned_weights_nl.npy', learned_weights)
    np.save('learned_biases_nl.npy', learned_biases)

    # plot the loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.show()

    # plot the learned terrain
    plt.figure()
    plt.plot(t, model.predict(x_train))
    plt.plot(t, data)
    plt.xlabel('Time')
    plt.ylabel('Terrain')
    plt.title('Learned Terrain vs Time')
    plt.show()
