import pickle
from src.simulator.animator import *
from src.geometries.live_origami_geom import LiveOrigamiFoldGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'live_origami_fold_test.pkl'
    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, a, xn, yn, dt] = params
    geom = LiveOrigamiFoldGeometry(a=a, xn=xn, yn=yn, visualize=True)

    simulate(geom, solution)
