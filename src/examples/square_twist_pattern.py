import pickle
from src.simulator.animator import *
from src.geometries.square_twist_geom import SquareTwistGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'SquareTwistRotation.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, l, phi, dt] = params
    geom = SquareTwistGeometry(l=l, phi=phi, visualize=True)

    simulate(geom, solution)
