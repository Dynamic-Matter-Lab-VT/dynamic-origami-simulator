from src.geometries.spring_geom import SpringGeometry
from src.models.spring_dynamic_model import *
import os

if __name__ == '__main__':
    d = 0.3
    p = 0.06
    n = 10
    th = 0.3

    zeta = 0.5
    # k_axial = 10000
    # k_shear = 500000
    k_axial = 100
    k_shear = 5000

    freq = 20.0

    geom = SpringGeometry(d=d, p=p, n=n, th=th, visualize=False, res=200)

    data_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../data/simulations/'
    # filename = data_dir + 'TaperedSpring_nlin_terrain_class' + str(freq) + '_hz.pkl'
    # filename = data_dir + 'TaperedSpring_lin_terrain_class' + str(freq) + '_hz.pkl'
    # filename = data_dir + 'TaperedSpring_linearity_test.pkl'
    # filename = data_dir + 'TaperedSpring_freq_analysis.pkl'
    # filename = data_dir + 'CylinderSpring_impulse_response.pkl'
    filename = data_dir + 'CylinderSpring_cubic_nonlinear.pkl'
    get_solution(geom, filename, zeta, k_axial, k_shear, 0.05, t_max_=50, freq_=freq)
