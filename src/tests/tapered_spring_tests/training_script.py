import numpy as np
import matplotlib.pyplot as plt
from src.tests.tapered_spring_tests.terrain_shape import *
import pickle
import tensorflow as tf

if __name__ == "__main__":
    tf.autograph.set_verbosity(3)
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../../data/simulations/'
    filename = data_dir + 'TaperedSpring_lin_terrain_class' + str(10.0) + '_hz.pkl'
    # filename = data_dir + 'TaperedSpring_nlin_terrain_class' + str(10.0) + '_hz.pkl'

    retrain = False

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    i_max = 100
    t = solution.t

    x = np.zeros((i_max, 3, t.shape[0]))
    for i in range(0, t.shape[0]):
        x[:, :, i] = solution.y[:i_max * 3, i].reshape((i_max, 3)) - solution.y[0:i_max * 3, 2].reshape((i_max, 3))
        # x[:, :, i] = solution.y[:i_max * 3, i].reshape((i_max, 3))

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
    readout = x[::readout_n, :, :] * 10

    load_data()
    data = get_terrian(t)

    plt.figure()
    plt.plot(t, data)
    plt.xlabel('Time')
    plt.ylabel('Terrain')
    plt.title('Terrain vs Time')
    plt.show()

    plt.figure()
    for i in range(readout.shape[0]):
        plt.plot(t, readout[i, 2, :])
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Displacement vs Time')
    plt.show()

    x_train = readout[:, 2, :].T
    y_train = data

    if retrain:
        weights = np.load('learned_weights_nl.npy')
        biases = np.load('learned_biases_nl.npy')
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1, input_shape=(x_train.shape[1],)))
        model.layers[0].set_weights([weights, biases])
        model.compile(optimizer='adam', loss='mean_squared_error')
    else:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1, input_shape=(x_train.shape[1],)))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

    print(x_train.shape, y_train.shape)
    history = model.fit(x_train, y_train, epochs=2000, verbose=1, batch_size=50)

    # save the weights and biases
    learned_weights, learned_biases = model.layers[0].get_weights()
    # save
    np.save('learned_weights_l.npy', learned_weights)
    np.save('learned_biases_l.npy', learned_biases)

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
