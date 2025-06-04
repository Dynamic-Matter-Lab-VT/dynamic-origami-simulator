from src.geometries.base_geom import BaseGeometry
from src.models.bar_hinge_model import *
import os

if __name__ == '__main__':
    a = 0.9
    b = 0.5
    cut_th = 0.001
    xn = 4
    yn = 6

    geom = BaseGeometry(a, b, xn, yn, False)
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'rubber_test.pkl'
    get_solution(geom, filename, 0.01, 20.0, 0.75, 0.8, 0.01, 20.0)
