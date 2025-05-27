from src.geometries.miura_ori_geom import MiuraOriGeometry
from src.models.bar_hinge_model import *
import os

if __name__ == '__main__':
    xn = 5
    yn = 5
    theta = 60 * np.pi / 180
    gamma = 50 * np.pi / 180
    a = 1.6
    b = 1.0

    geom = MiuraOriGeometry(a, b, gamma, theta, xn, yn, False)

    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'miura_ori_test.pkl'
    get_solution(geom, filename, 0.01, 20.0, 0.75, 0.8, 0.01, 20.0)
