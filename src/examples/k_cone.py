import pickle
from src.simulator.animator import *
from src.geometries.k_cone_geometry import KConeGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    k = 6  # number of sides
    filename = data_dir + str(k) + '_cone_test.pkl'
    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, k, l, h, th_c, hg_th, dt] = params
    geom = KConeGeometry(k=k, l=l, h=h, th_c=th_c, hg_th=hg_th, visualize=True)
    simulate(geom, solution)
