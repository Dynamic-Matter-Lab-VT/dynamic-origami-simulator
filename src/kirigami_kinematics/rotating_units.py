import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def transform(th_, x0_=np.array([0, 0, 1]).T):
    return np.array([[np.cos(th_), -np.sin(th_), x0_[0]],
                     [np.sin(th_), np.cos(th_), x0_[1]],
                     [0, 0, 1]])


def apply_reflection_x(sqs_):
    R = np.array([[-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    return [R @ sq_ for sq_ in sqs_]


def plot_sqs(unit_, ax_):
    sqs_, rect_, th_ = unit_
    for sq_ in sqs_:
        ax_.plot(sq_[0, :], sq_[1, :], 'b-')
        ax_.plot(sq_[0, :], sq_[1, :], 'ro')
    ax_.plot(rect_[0, :], rect_[1, :], 'g--')


def apply_transform(sqs_, R):
    return [R @ sq_ for sq_ in sqs_]


def get_unit0(sq_, th_, rebase=True):
    sq_r_ = apply_reflection_x(sq_)
    sq1 = apply_transform(sq_, transform(th_ / 2))
    sq2 = apply_transform(sq_r_, transform(-th_ / 2))
    sq3 = apply_transform(sq_r_, transform(th_ / 2, sq2[0][:, 1]))
    sq4 = apply_transform(sq_, transform(-th_ / 2, sq1[0][:, 1]))

    if rebase:
        sqs_ = apply_transform(sq1 + sq2 + sq3 + sq4, transform(0, -sq2[0][:, 3]))
    else:
        sqs_ = sq1 + sq2 + sq3 + sq4
    rect = np.array([sqs_[1][:, 3], sqs_[2][:, 2], sqs_[3][:, 2], sqs_[0][:, 3], sqs_[1][:, 3]]).T
    return sqs_, rect, th_


def get_next_unit(unit_, th_, rebase=True):
    sqs_p, rect_p, th_p = unit_
    th_ = th_ + th_p
    sup_rects_, new_rect_, th_ = get_unit0([rect_p], th_, rebase=False)

    sqs_p_r = apply_reflection_x(sqs_p)
    sq1 = apply_transform(sqs_p, transform(th_ / 2))
    sq2 = apply_transform(sqs_p_r, transform(-th_ / 2))
    sq3 = apply_transform(sqs_p_r, transform(th_ / 2, sup_rects_[1][:, 1]))
    sq4 = apply_transform(sqs_p, transform(-th_ / 2, sup_rects_[0][:, 1]))

    if rebase:
        sqs_ = apply_transform(sq1 + sq2 + sq3 + sq4, transform(0, -sup_rects_[1][:, 3]))
        new_rect_ = apply_transform([new_rect_], transform(0, -sup_rects_[1][:, 3]))[0]
    else:
        sqs_ = sq1 + sq2 + sq3 + sq4

    return sqs_, new_rect_, th_


if __name__ == "__main__":
    a = 1
    b = 1

    sq0 = np.array([[0, 0, 1],
                    [0, b, 1],
                    [-a, b, 1],
                    [-a, 0, 1],
                    [0, 0, 1]]).T

    th0 = np.pi / 6
    th1 = np.pi / 6

    unit0 = get_unit0([sq0], th0)
    unit1 = get_next_unit(unit0, th1)
    # unit2 = get_next_unit(unit1, np.pi / 6)

    fig, ax = plt.subplots()
    # plot_sqs(unit0, ax)
    plot_sqs(unit1, ax)
    # plot_sqs(unit2, ax)

    plt.axis('equal')
    plt.grid()
    plt.title("Rotating Square")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
