from src.geometries.spring_geom import SpringGeometry
from src.models.spring_dynamic_model import *
import os

if __name__ == '__main__':
    d = 0.3
    p = 0.06
    n = 10
    th = 0.0

    zeta = 50
    k_axial = 10000
    k_shear = 500000

    geom = SpringGeometry(d=d, p=p, n=n, th=th, visualize=False)

    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'SimpleSpring.pkl'
    get_solution(geom, filename, zeta, k_axial, k_shear, 0.01, 30)
