import numpy as np
# import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from benchmark_utils import input_func, narma
import os
from tqdm import tqdm
from nonliner_spring_mass_damper_chain import fourier_analysis, plot

data_dir = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))) + '/../data/nl_spring_mass_damper_raw_data/'


def linear_regression(data, t, num_readouts, target, epochs=1000, retrain=False, filename='test', save=False,
                      plot=False):
    num_steps = t.shape[0]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(num_readouts,)))
    if retrain:
        w = np.load(data_dir + filename + '_w.npy')
        b = np.load(data_dir + filename + '_b.npy')
        model.layers[0].set_weights([w, b])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # 60% train, 20% validation, 20% test
    train_size = int(num_steps * 0.6)
    val_size = int(num_steps * 0.2)

    x_train = data[:, :train_size]
    y_train = target[:train_size]

    x_val = data[:, train_size:train_size + val_size]
    y_val = target[train_size:train_size + val_size]

    x_test = data[:, train_size + val_size:]
    y_test = target[train_size + val_size:]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    history = model.fit(x_train.T, y_train, epochs=epochs, validation_data=(x_val.T, y_val), batch_size=50)

    # save the weights and biases
    learned_weights, learned_biases = model.layers[0].get_weights()
    print(learned_weights.shape, learned_biases.shape)
    # save
    if save:
        np.save(data_dir + filename + '_w.npy', learned_weights)
        np.save(data_dir + filename + '_b.npy', learned_biases)

    if plot:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend(['Train', 'Validation'])
        plt.show()

        # show training results
        plt.figure()
        plt.plot(t[:train_size], model.predict(x_train.T))
        plt.plot(t[:train_size], y_train)
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.title('Learned Response vs Time (Train)')
        plt.legend(['Predicted', 'Actual'])
        plt.show()

        # show validation results
        plt.figure()
        plt.plot(t[train_size:train_size + val_size], model.predict(x_val.T))
        plt.plot(t[train_size:train_size + val_size], y_val)
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.title('Learned Response vs Time (Validation)')
        plt.legend(['Predicted', 'Actual'])
        plt.show()

        # show the test results
        plt.figure()
        plt.plot(t[train_size + val_size:], model.predict(x_test.T))
        plt.plot(t[train_size + val_size:], y_test)
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.title('Learned Response vs Time (Test)')
        plt.legend(['Predicted', 'Actual'])
        plt.show()

    # return training loss, validation loss, test loss
    return history.history['loss'][-1], history.history['val_loss'][-1], model.evaluate(x_test.T, y_test)


def linear_regration_exact(data, target, filename='test', validation=False):
    if not validation:
        training_size = int(data.shape[1] * 0.8)
        test_size = data.shape[1] - training_size
        x = data[:, :training_size]
        y = target[:training_size]
        w_est = np.linalg.pinv((np.vstack([x, np.ones(x.shape[1])])).T) @ y

        w = w_est[:-1]
        b = w_est[-1]

        t_error = (np.sum((y - (w @ x + b)) ** 2) / training_size) ** 0.5
        v_error = (np.sum((target[training_size:] - (w @ data[:, training_size:] + b)) ** 2) / test_size) ** 0.5

        # save the weights and biases
        np.save(data_dir + filename + '_w.npy', w)
        np.save(data_dir + filename + '_b.npy', b)

        return t_error, v_error
    else:
        w = np.load(data_dir + filename + '_w.npy')
        b = np.load(data_dir + filename + '_b.npy')
        x = data
        y = target
        y_est = w @ x + b

        # plot
        plt.figure()
        plt.plot(y)
        plt.plot(y_est)
        plt.show()

        # test
        print(t.shape, data.shape)
        plt.figure()
        plt.plot(t, data.T)
        plt.show()

        return None, None


if __name__ == '__main__':
    bar = tqdm(total=100)
    # remove the csv file if it exists
    if os.path.exists(data_dir + 'losses_quintic.csv'):
        os.remove(data_dir + 'losses_quintic.csv')
    for exp_no in range(1, 101):
        narma_no = 2
        filename = 'quintic_' + str(exp_no)
        with open(data_dir + filename + '.pkl', 'rb') as f:
            data = pickle.load(f)

        t = data['t']
        y = data['y']
        n = data['n']
        ip = data['ip']

        target = narma(narma_no, 10000)[0]
        t_loss, v_loss = linear_regration_exact(y[:n, :], target, filename=filename + '_narma_' + str(narma_no))
        # t_loss, v_loss, test_loss = linear_regression(y[:n, :], t, n, target, epochs=500, retrain=False,
        #                                               filename=filename + '_narma_' + str(narma_no), save=True,
        #                                               plot=False)
        #
        with open(data_dir + 'losses_quintic.csv', 'a') as f:
            f.write(f'{exp_no},{narma_no},{t_loss},{v_loss}\n')
        bar.update(1)
    # read the number of experiments and loss data and plot
    data = np.genfromtxt(data_dir + 'losses_quintic.csv', delimiter=',')
    exps = data[:, 0]
    t_loss = data[:, 2]
    v_loss = data[:, 3]
    # test_loss = data[:, 4]
    plt.figure()
    plt.plot(exps, t_loss)
    plt.plot(exps, v_loss)
    # plt.plot(exps, test_loss)
    plt.legend(['Training Loss', 'Validation Loss', 'Test Loss'])
    plt.xlabel('Experiment')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Experiment')
    plt.show()

    # # read solution example
    with open(data_dir + 'quintic_96.pkl', 'rb') as f:
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
    target = narma(2, 10000)[0]

    linear_regration_exact(y[:n, :], target, filename='cubic_96_narma_2', validation=True)
