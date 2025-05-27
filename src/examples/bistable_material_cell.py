import pickle
from src.simulator.animator import *
from src.geometries.bistable_material_geom import BistableMaterialGeometry
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'BistableMaterialTest_diag_spr.pkl'
    # filename = data_dir + 'BistableMaterialTest_10_10_4.pkl'
    # filename = data_dir + 'bistable_material_test.pkl'
    with open(filename, 'rb') as f:
        solution = pickle.load(f)

    params = solution.sol
    [geometryType, a, xn, yn, dt] = params
    geom = BistableMaterialGeometry(a=a, xn=xn, yn=yn, visualize=True)

    simulate(geom, solution)
