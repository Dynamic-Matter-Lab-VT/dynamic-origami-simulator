"""
Dynamic Model Solver for bistable materials

This code models bistable material structures defined in the geometry classes into a reduced bar-hinge model.
It solves the equations of motion for the origami structure and saves the simulation results.

Author: Yogesh Phalak
Date: 2023-08-29
Filename: BistableMaterialDynamicModel.py

"""

import numpy as np
import pickle
from scipy.integrate import solve_ivp
from numba import jit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from src.geometries.bistable_material_geom import BistableMaterialGeometry
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
    node_props[i_max // 2][j_max // 2][1] = True
    node_props[i_max // 2][j_max // 2 - 1][1] = True
    node_props[i_max // 2 - 1][j_max // 2][1] = True
    node_props[i_max // 2 - 1][j_max // 2 - 1][1] = True


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
    if 0.0 < t < 0.5:
        for i in range(i_max):
            force_external[i][0] = np.array([0.0, 0.0, 0.2])
            force_external[i][j_max - 1] = np.array([0.0, 0.0, 0.2])
    elif 5.5 < t < 6.5:
        for j in range(j_max):
            force_external[0][j] = np.array([0.0, 0.0, -0.2])
            force_external[i_max - 1][j] = np.array([0.0, 0.0, -0.2])
    else:
        for i in range(i_max):
            for j in prange(j_max):
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
        [p0, p1, l0] = bars[i]
        n = nodes[p1[0]][p1[1]] - nodes[p0[0]][p0[1]]
        l_cr = np.linalg.norm(n)
        n = n / l_cr
        c = 2 * zeta * np.sqrt(k_axial * l0)
        if l0 > a:
            k = k_str_spring
        else:
            k = k_axial
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

    for i in prange(len(vhinges)):
        vhinge = vhinges[i]
        hhinge = hhinges[i]

        [pjv, pkv, piv, plv, th0v, l0v, typv] = vhinge
        [pjh, pkh, pih, plh, th0h, l0h, typh] = hhinge

        r_ijv = get_vector(piv, pjv, nodes)
        r_kjv = get_vector(pkv, pjv, nodes)
        r_klv = get_vector(pkv, plv, nodes)

        r_ijh = get_vector(pih, pjh, nodes)
        r_kjh = get_vector(pkh, pjh, nodes)
        r_klh = get_vector(pkh, plh, nodes)

        mv = np.cross(r_ijv, r_kjv)
        nv = np.cross(r_kjv, r_klv)

        mh = np.cross(r_ijh, r_kjh)
        nh = np.cross(r_kjh, r_klh)

        thv = get_angle(mv, nv, r_klv)
        thh = get_angle(mh, nh, r_klh)

        if thh <= th0h:
            f_crh = -k_fold * l0h * (thv - th0v) ** 2 * (thh - th0h)
        elif thh > th0h:
            f_crh = -k_fold * l0h * (thv - th0v) ** 2 * (thh - th0h) * 100

        if thv <= th0v:
            f_crv = -k_fold * l0v * (thh - th0h) ** 2 * (thv - th0v)
        elif thv > th0v:
            f_crv = -k_fold * l0v * (thh - th0h) ** 2 * (thv - th0v) * 100

        dth_dxiv = np.linalg.norm(r_kjv) / np.linalg.norm(mv) ** 2 * mv
        dth_dxlv = -np.linalg.norm(r_kjv) / np.linalg.norm(nv) ** 2 * nv
        dth_dxjv = (np.dot(r_ijv, r_kjv) / np.linalg.norm(r_kjv) ** 2 - 1) * dth_dxiv \
                   - np.dot(r_klv, r_kjv) / np.linalg.norm(r_kjv) ** 2 * dth_dxlv
        dth_dxkv = (np.dot(r_klv, r_kjv) / np.linalg.norm(r_kjv) ** 2 - 1) * dth_dxlv \
                   - np.dot(r_ijv, r_kjv) / np.linalg.norm(r_kjv) ** 2 * dth_dxiv

        dth_dxih = np.linalg.norm(r_kjh) / np.linalg.norm(mh) ** 2 * mh
        dth_dxlh = -np.linalg.norm(r_kjh) / np.linalg.norm(nh) ** 2 * nh
        dth_dxjh = (np.dot(r_ijh, r_kjh) / np.linalg.norm(r_kjh) ** 2 - 1) * dth_dxih \
                   - np.dot(r_klh, r_kjh) / np.linalg.norm(r_kjh) ** 2 * dth_dxlh
        dth_dxkh = (np.dot(r_klh, r_kjh) / np.linalg.norm(r_kjh) ** 2 - 1) * dth_dxlh \
                   - np.dot(r_ijh, r_kjh) / np.linalg.norm(r_kjh) ** 2 * dth_dxih

        if not node_props[piv[0]][piv[1]][1]:
            force_crease[piv[0]][piv[1]] += f_crv * dth_dxiv
        if not node_props[pjv[0]][pjv[1]][1]:
            force_crease[pjv[0]][pjv[1]] += f_crv * dth_dxjv
        if not node_props[pkv[0]][pkv[1]][1]:
            force_crease[pkv[0]][pkv[1]] += f_crv * dth_dxkv
        if not node_props[plv[0]][plv[1]][1]:
            force_crease[plv[0]][plv[1]] += f_crv * dth_dxlv

        if not node_props[pih[0]][pih[1]][1]:
            force_crease[pih[0]][pih[1]] += f_crh * dth_dxih
        if not node_props[pjh[0]][pjh[1]][1]:
            force_crease[pjh[0]][pjh[1]] += f_crh * dth_dxjh
        if not node_props[pkh[0]][pkh[1]][1]:
            force_crease[pkh[0]][pkh[1]] += f_crh * dth_dxkh
        if not node_props[plh[0]][plh[1]][1]:
            force_crease[plh[0]][plh[1]] += f_crh * dth_dxlh


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


def get_solution(geom_, filename_='SimpleSpring.pkl', zeta_=0.01, k_axial_=5.0, k_facet_=0.1, k_fold_=15.0, k_str_spring_=0.75,
                 dt_=0.01, t_max_=10.0):
    global geom, filename, zeta, k_axial, k_facet, k_fold, k_str_spring, dt, t_max, i_max, j_max, node_props, bars, \
        hinges, vhinges, hhinges, force_external, force_axial, force_crease, force_damping, nodes0, vel0, t_sim, x0, \
        computation_progress, pbar, a, xn, yn

    geom = geom_
    filename = filename_
    zeta = zeta_
    k_axial = k_axial_
    k_facet = k_facet_
    k_fold = k_fold_
    k_str_spring = k_str_spring_
    dt = dt_
    t_max = t_max_

    a = geom.a
    xn = geom.xn
    yn = geom.yn

    i_max = geom.i_max
    j_max = geom.j_max
    node_props = [[[0.01, False] for j in range(0, j_max)] for i in range(0, i_max)]
    bars = geom.bars
    hinges = geom.hinges
    vhinges = geom.vhinges
    hhinges = geom.hhinges

    force_external = np.zeros((i_max, j_max, 3))
    force_axial = np.zeros((i_max, j_max, 3))
    force_crease = np.zeros((i_max, j_max, 3))
    force_damping = np.zeros((i_max, j_max, 3))

    nodes0 = geom.nodes
    vel0 = np.zeros((i_max, j_max, 3))

    t_sim = np.linspace(0, t_max, int(t_max / dt) + 1)
    x0 = np.hstack((nodes0.reshape(i_max * j_max * 3), vel0.reshape(i_max * j_max * 3)))

    print(t_sim)

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
    a = 0.5
    xn = 5
    yn = 5

    geom = BistableMaterialGeometry(a, xn, yn, False)
    filename = 'SimpleSpring.pkl'
    get_solution(geom, filename, 0.01, 20.0, 0.75, 0.8, 0.01, 20.0)
