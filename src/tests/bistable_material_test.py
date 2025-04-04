from src.geometries.bistable_material_geom import BistableMaterialGeometry
from src.models.bistable_material_model import *
import os

if __name__ == '__main__':
    a = 0.5
    xn = 10
    yn = 10

    geom = BistableMaterialGeometry(a, xn, yn, False)

    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    filename = data_dir + 'bistable_material_test.pkl'
    get_solution(geom, filename, 0.01, 20.0, 0.75, 0.8, 0.01, 0.01, 10.0)
