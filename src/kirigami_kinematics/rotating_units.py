import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def transform(th_, x0_=np.array([0, 0, 1]).T):
    return np.array([[np.cos(th_), np.sin(th_), x0_[0]],
                     [-np.sin(th_), np.cos(th_), x0_[1]],
                     [0, 0, 1]])


def plot_sq(sq_, ax_):
    ax.plot(sq_[0, :], sq_[1, :], 'b-')
    ax.plot(sq_[0, :], sq_[1, :], 'ro')


def apply_transform(sqs_, R):
    return [R @ sq_ for sq_ in sqs_]


def unit(sq_, th_):
    sq1 = apply_transform([sq_], transform(th_ / 2 + np.pi / 2))[0]
    sq2 = apply_transform([sq_], transform(-th_ / 2 + np.pi / 2, sq1[:, 3]))[0]
    sq4 = apply_transform([sq_], transform(-th_ / 2))[0]
    sq3 = apply_transform([sq_], transform(th_ / 2, sq4[:, 1]))[0]

    sq1, sq2, sq3, sq4 = apply_transform([sq1, sq2, sq3, sq4], transform(0, -sq1[:, 1]))

    return sq1, sq2, sq3, sq4


def unit2(u0_, th_):
    u1 = apply_transform(u0_, transform(th_ / 2 + np.pi / 2))
    u2 = apply_transform(u0_, transform(-th_ / 2, u1[0][:, 1]))
    u4 = apply_transform(u0_, transform(-th_ / 2))
    u3 = apply_transform(u0_, transform(th_ / 2, u4[0][:, 1]))

    # u1, u2, u3, u4 = apply_transform([u1[0], u2[0], u3[0], u4[0]], transform(0, -u1[0][:, 1]))
    return u1, u2, u3, u4


if __name__ == "__main__":
    sq0 = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [-1, 1, 1],
                    [-1, 0, 1],
                    [0, 0, 1]]).T
    unit0 = unit(sq0, np.pi / 3)
    unit1 = unit2(unit0, np.pi / 6)

    fig, ax = plt.subplots()
    # plot_sq(unit0[0], ax)
    # plot_sq(unit0[1], ax)
    # plot_sq(unit0[2], ax)
    # plot_sq(unit0[3], ax)

    print(unit1[0])

    plot_sq(unit1[0][0], ax)
    plot_sq(unit1[0][1], ax)
    plot_sq(unit1[0][2], ax)
    plot_sq(unit1[0][3], ax)

    plot_sq(unit1[1][0], ax)
    plot_sq(unit1[1][1], ax)
    plot_sq(unit1[1][2], ax)
    plot_sq(unit1[1][3], ax)

    # plt.plot(unit0[0][0, 1], unit0[0][1, 1], 'k*')
    # plt.plot(unit0[1][0, 2], unit0[1][1, 2], 'k*')
    # plt.plot(unit0[2][0, 2], unit0[2][1, 2], 'k*')
    # plt.plot(unit0[3][0, 3], unit0[3][1, 3], 'k*')

    plt.axis('equal')
    plt.grid()
    plt.title("Rotating Square")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
