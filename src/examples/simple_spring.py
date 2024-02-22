import pickle
from src.simulator.spring_sim import *
from src.geometries.spring_geom import SpringGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'SimpleSpring.pkl'
    # filename = data_dir + 'TaperedSpring_0.pkl'
    # filename = data_dir + 'TaperedSpring_50_hz.pkl'
    # freq = str(input())
    freq = '8.734_'
    filename = data_dir + 'TaperedSpring_sq' + freq + 'hz.pkl'

    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    res = solution.y.shape[0] // 6
    [geometryType, d, p, n, th, dt] = params
    geom = SpringGeometry(d=d, p=p, n=n, th=th, visualize=True, res=res)

    simulate(geom, solution)
