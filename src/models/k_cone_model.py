"""
Dynamic Model Solver for bistable materials

This code models bistable material structures defined in the geometry classes into a reduced bar-hinge model.
It solves the equations of motion for the origami structure and saves the simulation results.

Author: Yogesh Phalak
Date: 2025-10-01
Filename: KConeModel.py

"""

import numpy as np
import pickle
from scipy.integrate import solve_ivp
from numba import jit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from src.geometries.k_cone_geometry import KConeGeometry
from tqdm import tqdm

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


@jit(fastmath=True, cache=True)
def get_angle(m, n, rkl):
    """
        Calculate the angle between two vectors in 3D space.

        Args:
            m (numpy.ndarray): The first vector.
            n (numpy.ndarray): The second vector.
            rkl (numpy.ndarray): Vector perpendicular to first and second.

        Returns:
            float: The angle between vectors m and n.

    """
    if np.linalg.norm(m) == 0 or np.linalg.norm(n) == 0:
        return 0
    else:
        c_th = max(min(np.dot(m, n) / (np.linalg.norm(m) * np.linalg.norm(n)), 1.0), -1.0)
        if np.dot(m, rkl) == 0.0:
            return np.arccos(c_th) % (2 * np.pi)
        else:
            return (np.sign(np.dot(m, rkl)) * np.arccos(c_th)) % (2 * np.pi)


@jit(fastmath=True, cache=True)
def get_vector(pa, pb, nodes):
    """
        Calculate the vector between two nodes.

        Args:
            pa (list): Coordinates of the first node.
            pb (list): Coordinates of the second node.
            nodes (numpy.ndarray): Array of node coordinates.

        Returns:
            numpy.ndarray: The vector between nodes pa and pb.

    """
    return nodes[pa[0]][pa[1]] - nodes[pb[0]][pb[1]]


def update_node_properties():
    """
       Update the properties of the nodes.

       This function updates the properties of the nodes (mass and fixed) based on certain conditions.

    """
    global node_props, i_max, j_max
    n = i_max // 4
    for i in range(n):
        node_props[(4 * i + 1) % (4 * n)][0][1] = True


def initialize_forces():
    """
        Initialize the force arrays.

        This function initializes the force arrays.

    """
    global i_max, j_max, force_external, force_axial, force_crease, force_damping
    force_external, force_axial, force_crease, force_damping = [np.zeros((i_max, j_max, 3)),
                                                                np.zeros((i_max, j_max, 3)),
                                                                np.zeros((i_max, j_max, 3)),
                                                                np.zeros((i_max, j_max, 3))]


# @jit(nopython=True)
def calculate_external_force(nodes, vel, t):
    """
        Calculate the external forces acting on the nodes.

        Args:
            nodes (numpy.ndarray): Array of node coordinates.
            vel (numpy.ndarray): Array of node velocities.
            t (float): Time value.

    """
    global i_max, j_max, node_props, force_external
    # if 0.0 < t < 0.5:
    #     for i in range(i_max):
    #         force_external[i][0] = np.array([0.0, 0.0, 0.2])
    #         force_external[i][j_max - 1] = np.array([0.0, 0.0, 0.2])
    # elif 5.5 < t < 6.5:
    #     for j in range(j_max):
    #         force_external[0][j] = np.array([0.0, 0.0, -0.2])
    #         force_external[i_max - 1][j] = np.array([0.0, 0.0, -0.2])
    # else:
    #     for i in range(i_max):
    #         for j in prange(j_max):
    #             force_external[i][j] = np.array([0.0, 0.0, 0.0])

    for i in range(i_max):
        for j in range(j_max):
            if node_props[i][j][1]:
                force_external[i][j] = np.array([0.0, 0.0, 0.0])
            else:
                force_external[i][j] = np.array([0.0, 0.0, 0.0])


# @jit(parallel=True, cache=True, fastmath=True)
def calculate_axial_force(nodes, vel, t):
    """
        Calculate the axial forces acting on the bars.

        Args:
            nodes (numpy.ndarray): Array of node coordinates.
            vel (numpy.ndarray): Array of node velocities.
            t (float): Time value.

    """
    global k_axial, node_props, bars, zeta, force_axial, force_damping, a, k_str_spring

    for i in prange(0, len(bars)):
        if len(bars[i]) == 3:
            [p0, p1, l0] = bars[i]
            typ = None
        else:
            p0 = bars[i][0]
            p1 = bars[i][1]
            l0 = bars[i][2]
            typ = bars[i][3]

        if typ == 'hydrogel':
            l0 = l0 * (1.6 + 0.9 * np.sin(2 * np.pi * t / 3.0))
            # if t < 1.0:
            #     l0 = l0 * 1.5
            # elif t < 2.0:
            #     l0 = l0 * 0.7
            # elif t < 3.0:
            #     l0 = l0 * 1.5
            k = k_str_spring
        else:
            k = k_axial

        n = nodes[p1[0]][p1[1]] - nodes[p0[0]][p0[1]]
        l_cr = np.linalg.norm(n)
        n = n / l_cr
        c = 2 * zeta * np.sqrt(k_axial * l0)

        if not node_props[p0[0]][p0[1]][1]:
            force_axial[p0[0]][p0[1]] += k * (l_cr - l0) / l0 * n
            force_damping[p0[0]][p0[1]] += c * (vel[p1[0]][p1[1]] - vel[p0[0]][p0[1]])
        if not node_props[p1[0]][p1[1]][1]:
            force_axial[p1[0]][p1[1]] -= k * (l_cr - l0) / l0 * n
            force_damping[p1[0]][p1[1]] += c * (vel[p0[0]][p0[1]] - vel[p1[0]][p1[1]])


# @jit(cache=True, fastmath=True, parallel=True)
def calculate_crease_force(nodes, vel, t):
    """
       Calculate the crease forces acting on the hinges.

       Args:
           nodes (numpy.ndarray): Array of node coordinates.
           vel (numpy.ndarray): Array of node velocities.
           t (float): Time value.

    """
    global k_fold, k_facet, node_props, hinges, hhinges, vhinges, force_crease
    for i in prange(len(hinges)):
        hinge = hinges[i]
        [pj, pk, pi, pl, th0, l0, typ] = hinge

        r_ij = get_vector(pi, pj, nodes)
        r_kj = get_vector(pk, pj, nodes)
        r_kl = get_vector(pk, pl, nodes)

        m = np.cross(r_ij, r_kj)
        n = np.cross(r_kj, r_kl)

        th = get_angle(m, n, r_kl)

        if typ == 'facet':
            k = k_facet * l0
        else:
            k = k_fold * l0

        dth_dxi = np.linalg.norm(r_kj) / np.linalg.norm(m) ** 2 * m
        dth_dxl = -np.linalg.norm(r_kj) / np.linalg.norm(n) ** 2 * n
        dth_dxj = (np.dot(r_ij, r_kj) / np.linalg.norm(r_kj) ** 2 - 1) * dth_dxi \
                  - np.dot(r_kl, r_kj) / np.linalg.norm(r_kj) ** 2 * dth_dxl
        dth_dxk = (np.dot(r_kl, r_kj) / np.linalg.norm(r_kj) ** 2 - 1) * dth_dxl \
                  - np.dot(r_ij, r_kj) / np.linalg.norm(r_kj) ** 2 * dth_dxi

        f_cr = -k * (th - th0)

        if not node_props[pi[0]][pi[1]][1]:
            force_crease[pi[0]][pi[1]] += f_cr * dth_dxi
        if not node_props[pj[0]][pj[1]][1]:
            force_crease[pj[0]][pj[1]] += f_cr * dth_dxj
        if not node_props[pk[0]][pk[1]][1]:
            force_crease[pk[0]][pk[1]] += f_cr * dth_dxk
        if not node_props[pl[0]][pl[1]][1]:
            force_crease[pl[0]][pl[1]] += f_cr * dth_dxl


# @jit(cache=True, fastmath=True)
def bar_hinge_model(t, x):
    """
        Compute the time derivatives of the state variables.

        Args:
            t (float): Time value.
            x (numpy.ndarray): Array of state variables.

        Returns:
            numpy.ndarray: Array of time derivatives of the state variables.

    """
    global force_external, force_axial, force_crease, force_damping, node_props, computation_progress, pbar

    if t == 0.0:
        pbar = tqdm(total=100, desc='Computation progress', colour='blue', unit='%')
    else:
        pbar.update((t / t_max - computation_progress) * 100)
        computation_progress = t / t_max

    initialize_forces()
    nodes = x[:i_max * j_max * 3].reshape((i_max, j_max, 3))
    vel = x[i_max * j_max * 3:].reshape((i_max, j_max, 3))

    calculate_external_force(nodes, vel, t)
    calculate_crease_force(nodes, vel, t)
    calculate_axial_force(nodes, vel, t)

    acc = (force_external + force_axial + force_crease + force_damping) / np.repeat(
        (np.array(node_props)[:, :, 0])[:, :, np.newaxis], 3, axis=2)

    dx_dt = np.zeros(i_max * j_max * 6)
    dx_dt[:i_max * j_max * 3] = vel.reshape(i_max * j_max * 3)
    dx_dt[i_max * j_max * 3:] = acc.reshape(i_max * j_max * 3)

    return dx_dt


def get_solution(geom_, filename_='SimpleSpring.pkl', zeta_=0.01, k_axial_=5.0, k_facet_=0.1, k_fold_=15.0,
                 k_str_spring_=0.75,
                 dt_=0.01, t_max_=10.0):
    global geom, filename, zeta, k_axial, k_facet, k_fold, k_str_spring, dt, t_max, i_max, j_max, node_props, bars, \
        hinges, vhinges, hhinges, force_external, force_axial, force_crease, force_damping, nodes0, vel0, t_sim, x0, \
        computation_progress, pbar

    geom = geom_
    filename = filename_
    zeta = zeta_
    k_axial = k_axial_
    k_facet = k_facet_
    k_fold = k_fold_
    k_str_spring = k_str_spring_
    dt = dt_
    t_max = t_max_

    i_max = geom.i_max
    j_max = geom.j_max
    node_props = [[[0.01, False] for j in range(0, j_max)] for i in range(0, i_max)]
    bars = geom.bars
    hinges = geom.hinges

    force_external = np.zeros((i_max, j_max, 3))
    force_axial = np.zeros((i_max, j_max, 3))
    force_crease = np.zeros((i_max, j_max, 3))
    force_damping = np.zeros((i_max, j_max, 3))

    nodes0 = geom.nodes
    vel0 = np.zeros((i_max, j_max, 3))

    t_sim = np.linspace(0, t_max, int(t_max / dt) + 1)
    x0 = np.hstack((nodes0.reshape(i_max * j_max * 3), vel0.reshape(i_max * j_max * 3)))

    computation_progress = 0
    pbar = None

    update_node_properties()
    x_sol = solve_ivp(fun=bar_hinge_model, t_span=(0, t_max), y0=x0, method='RK23', t_eval=t_sim)
    x_sol.sol = geom.params + [dt]
    x_sol.y_events = node_props
    pbar.close()
    print(x_sol)

    with open(filename, 'wb') as f:
        pickle.dump(x_sol, f)


if __name__ == '__main__':
    geom = KConeGeometry(visualize=False)
    filename = 'KCone.pkl'
    get_solution(geom, filename, 0.01, 20.0, 0.75, 0.0, 0.01, 20.0)
