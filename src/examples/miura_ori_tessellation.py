import pickle
from src.simulator.animator import *
from src.geometries.miura_ori_geom import MiuraOriGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'MiuraOri_5_5_gravity.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, a, b, gamma, theta, xn, yn, dt] = params
    geom = MiuraOriGeometry(a=a, b=b, gamma=gamma, theta=theta, xn=xn, yn=yn, visualize=True)

    simulate(geom, solution)
