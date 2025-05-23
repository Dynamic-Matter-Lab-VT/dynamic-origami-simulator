from src.geometries.live_origami_geom import LiveOrigamiFoldGeometry
from src.models.live_origami_fold_model import *
import os

if __name__ == '__main__':
    a = 0.5
    xn = 3
    yn = 6

    geom = LiveOrigamiFoldGeometry(a, xn, yn, False)

    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'live_origami_fold_test.pkl'
    filename = data_dir + 'live_origami_fold_sq_test.pkl'
    get_solution(geom, filename, 0.01, 20.0, 10.0, 0.1, 5.0, 0.01, 10.0)
