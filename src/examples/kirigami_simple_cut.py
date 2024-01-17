import pickle
from src.simulator.animator import *
from src.geometries.simple_cut_geom import KirigamiSimpleCutGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'KirigamiSimpleCut_4_6_PlanerForce.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, a, b, cut_th, xn, yn, dt] = params
    geom = KirigamiSimpleCutGeometry(a=a, b=b, cut_th=cut_th, xn=xn, yn=yn, visualize=True)

    simulate(geom, solution)
