import pickle
from src.simulator.spring_sim import *
from src.geometries.spring_geom import SpringGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'SimpleSpring.pkl'
    filename = data_dir + 'TaperedSpring.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, d, p, n, th, dt] = params
    geom = SpringGeometry(d=d, p=p, n=n, th=th, visualize=True)

    simulate(geom, solution)
