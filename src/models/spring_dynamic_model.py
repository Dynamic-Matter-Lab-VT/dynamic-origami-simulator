import numpy as np
from numba import jit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from src.geometries.spring_geom import SpringGeometry
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pickle

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


@jit(fastmath=True, cache=True)
def update_node_props():
    global node_props, i_max, n
    for i in range(i_max // n):
        node_props[i][1] = True


@jit(fastmath=True, cache=True)
def initialize_forces():
    global force_external, force_axial, force_shear, force_damping, i_max
    force_external[:, :] = np.zeros((i_max, 3))
    force_axial[:, :] = np.zeros((i_max, 3))
    force_shear[:, :] = np.zeros((i_max, 3))
    force_damping[:, :] = np.zeros((i_max, 3))


@jit(fastmath=True, cache=True)
def calculate_external_force(x, x_d, t):
    global force_external, node_props, i_max
    for i in prange(i_max):
        if not node_props[i][1]:
            force_external[i, 0] = 0
            force_external[i, 1] = 0
            force_external[i, 2] = -9.81 * node_props[i][0]


@jit(fastmath=True, cache=True)
def calculate_internal_force(x, x_d, t):
    global force_axial, force_shear, force_damping, node_props, i_max, k_axial, k_shear, zeta
    for i in prange(i_max - 1):
        l0 = np.linalg.norm(x0[i + 1] - x0[i])
        l = np.linalg.norm(x[i + 1] - x[i])
        r0 = x0[i + 1, :] - x0[i, :]
        r = x[i + 1, :] - x[i, :]
        if not node_props[i][1]:
            force_axial[i, :] += k_axial * (l - l0) * (x[i + 1, :] - x[i, :]) / l
            force_damping[i, :] += zeta * (x_d[i + 1, :] - x_d[i, :])
            force_shear[i, :] += k_shear * (r - r0)
        if not node_props[i + 1][1]:
            force_axial[i + 1, :] -= k_axial * (l - l0) * (x[i + 1, :] - x[i, :]) / l
            force_damping[i + 1, :] -= zeta * (x_d[i + 1, :] - x_d[i, :])
            force_shear[i + 1, :] += -k_shear * (r - r0)


def dynamic_model(t, z):
    global force_external, force_axial, force_shear, force_damping, node_props, t_max, pbar, computation_progress
    if t == 0:
        pbar = tqdm(total=100, desc='Simulation Progress')
    else:
        pbar.update((t / t_max - computation_progress) * 100)
        computation_progress = t / t_max

    x = z[0:i_max * 3].reshape((i_max, 3))
    x_d = z[i_max * 3:].reshape((i_max, 3))
    x_dd = np.zeros((i_max, 3))

    initialize_forces()
    calculate_external_force(x, x_d, t)
    calculate_internal_force(x, x_d, t)

    for i in range(i_max):
        x_dd[i, :] = (force_external[i, :] + force_axial[i, :] + force_shear[i, :] + force_damping[i, :]) / \
                     node_props[i][0]

    dz_dt = np.concatenate((x_d.flatten(), x_dd.flatten()))
    return dz_dt


def get_solution(geom_, filename_='SimpleSpring.pkl', zeta_=30, k_axial_=10000, k_shear_=10000, dt_=0.01, t_max_=30):
    global geom, zeta, k_axial, k_shear, dt, x, x0, i_max, force_external, force_axial, force_shear, force_damping, \
        node_props, t_sim, computation_progress, pbar, z0, t_max, filename, n

    filename = filename_

    geom = geom_
    zeta = zeta_
    k_axial = k_axial_
    k_shear = k_shear_

    dt = dt_
    x = geom.x
    x0 = geom.x0
    i_max = geom.i_max
    n = geom.n

    force_external = np.zeros((i_max, 3))
    force_axial = np.zeros((i_max, 3))
    force_shear = np.zeros((i_max, 3))
    force_damping = np.zeros((i_max, 3))

    node_props = [[0.1, False] for i in range(i_max)]
    update_node_props()

    t_max = t_max_
    t_sim = np.arange(0, t_max, dt)
    computation_progress = 0
    pbar = None

    z0 = np.concatenate((x0.flatten(), np.zeros((i_max, 3)).flatten()))

    sol = solve_ivp(dynamic_model, [0, t_max], z0, t_eval=t_sim, method='RK45')
    sol.sol = geom.params + [dt]
    sol.y_events = node_props
    with open(filename, 'wb') as f:
        pickle.dump(sol, f)


if __name__ == '__main__':
    d = 0.6
    p = 0.1
    th = 0.0
    n = 5
    geom = SpringGeometry(d, p, n, th, False)
    get_solution(geom, filename_='SimpleSpring.pkl', zeta_=30, k_axial_=10000, k_shear_=10000, dt_=0.01, t_max_=30)
