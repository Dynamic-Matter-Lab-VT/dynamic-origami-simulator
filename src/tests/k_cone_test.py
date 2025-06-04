from src.geometries.k_cone_geometry import KConeGeometry
from src.models.k_cone_model import *
import os

if __name__ == '__main__':
    k = 6  # number of sides
    l = 1.0  # length of the side base
    h = 0.1  # height of the hole
    hg_th = 0.1  # thickness of the hydrogel
    th_c = 0.05  # angle of the cut size

    geom = KConeGeometry(k=k, l=l, h=h, th_c=th_c, hg_th=hg_th, visualize=False)

    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + str(k) + '_cone_test.pkl'
    get_solution(geom, filename, 0.01, 20.0, 10.0, 0.01, 5.0, 0.01, 10.0)
