import pickle
from src.simulator.animator import *
from src.geometries.base_geom import BaseGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'RubberSheet.pkl'
    filename = data_dir + 'test.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, a, b, xn, yn, dt] = params
    geom = BaseGeometry(a=a, b=b, xn=xn, yn=yn, visualize=True)

    simulate(geom, solution)
