import numpy as np
import cv2
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from numba import jit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from tqdm import tqdm
import time


class BioGrid:
    def __init__(self, m=4, n=4):
        self.m = m
        self.n = n
        self.cells = None
        self.init_cells()
        self.nodes = np.zeros((2 * m + 1, 2 * n + 1, 2))
        self.fixed = np.zeros((2 * m + 1, 2 * n + 1), dtype=bool)
        self.acceleration = np.zeros_like(self.nodes)
        self.velocity = np.zeros_like(self.nodes)
        self.l = 500 // np.max(self.nodes.shape)
        self.bars = []
        self.init_bars()
        self.canvas = np.zeros(((self.nodes.shape[1] + 1) * self.l, (self.nodes.shape[0] + 1) * self.l, 3),
                               dtype=np.uint8)
        self.grabbed = False
        self.grabbed_i = 0
        self.grabbed_j = 0
        self.t0 = time.time()
        self.t = time.time() - self.t0
        self.fps = 0
        cv2.namedWindow('canvas')
        cv2.setMouseCallback('canvas', self.mouse_callback)

    def init_nodes(self):
        for i in range(self.nodes.shape[0]):
            for j in range(self.nodes.shape[1]):
                self.nodes[i, j] = [(i + 1) * self.l, (j + 1) * self.l]

        # fix corners
        self.fixed[0, 0] = True
        self.fixed[0, -1] = True
        self.fixed[-1, 0] = True
        self.fixed[-1, -1] = True

    def apply_forces(self):
        self.acceleration = np.zeros_like(self.nodes)
        for bar in self.bars:
            p0, p1, l0 = bar
            f = (l0 * self.l - np.linalg.norm(self.nodes[p0[0], p0[1]] - self.nodes[p1[0], p1[1]])) * 1500

            self.acceleration[p0[0], p0[1]] += f * (
                    self.nodes[p0[0], p0[1]] - self.nodes[p1[0], p1[1]]) / np.linalg.norm(
                self.nodes[p0[0], p0[1]] - self.nodes[p1[0], p1[1]])
            # apply damping
            self.acceleration[p0[0], p0[1]] += -2.5 * self.velocity[p0[0], p0[1]]
            self.acceleration[p1[0], p1[1]] -= f * (
                    self.nodes[p0[0], p0[1]] - self.nodes[p1[0], p1[1]]) / np.linalg.norm(
                self.nodes[p0[0], p0[1]] - self.nodes[p1[0], p1[1]])
            # apply damping
            self.acceleration[p1[0], p1[1]] += -2.5 * self.velocity[p1[0], p1[1]]

        # self.acceleration += -2 * self.velocity

        self.acceleration[self.fixed] = 0

    def init_bars(self):
        for i in prange(self.nodes.shape[0]):
            for j in prange(self.nodes.shape[1]):
                if (i % 2 + j % 2) == 2:
                    if self.cells[(i - 1) // 2, (j - 1) // 2] == 1:
                        self.bars.append([[i, j], [i + 1, j], 0.5])
                        self.bars.append([[i, j], [i, j + 1], 0.5])
                        self.bars.append([[i, j], [i - 1, j], 0.5])
                        self.bars.append([[i, j], [i, j - 1], 0.5])

                else:
                    if ((i + 1) % 2 + j % 2) == 2:
                        pass
                    elif i < self.nodes.shape[0] - 1:
                        self.bars.append([[i, j], [i + 1, j], 1])
                    if (i % 2 + (j + 1) % 2) == 2:
                        pass
                    elif j < self.nodes.shape[1] - 1:
                        self.bars.append([[i, j], [i, j + 1], 1])

    def init_cells(self):
        self.cells = np.zeros((self.m, self.n))
        for i in prange(self.m):
            for j in prange(self.n):
                if np.random.rand() < 0.25:
                    self.cells[i, j] = 1

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.grabbed = not self.grabbed
            # self.grabbed = False
            for i in range(self.nodes.shape[0]):
                for j in range(self.nodes.shape[1]):
                    if np.linalg.norm(np.array([x, y]) - self.nodes[i, j]) < 10:
                        self.grabbed_i = i
                        self.grabbed_j = j
                        print('grabbed', i, j)
                        break

        if self.grabbed:
            self.nodes[self.grabbed_i, self.grabbed_j] = np.array([x, y])

    def draw(self):
        self.canvas.fill(255)

        for bar in self.bars:
            cv2.line(self.canvas, (int(self.nodes[bar[0][0], bar[0][1]][0]), int(self.nodes[bar[0][0], bar[0][1]][1])),
                     (int(self.nodes[bar[1][0], bar[1][1]][0]), int(self.nodes[bar[1][0], bar[1][1]][1])), (0, 255, 0),
                     2)

        for i in prange(self.nodes.shape[0]):
            for j in prange(self.nodes.shape[1]):
                if (i % 2 + j % 2) == 0:
                    # cv2.circle(self.canvas, (int(self.nodes[i, j][0]), int(self.nodes[i, j][1])), 5, (0, 0, 0), -1)
                    pass
                elif (i % 2 + j % 2) == 1:
                    # cv2.circle(self.canvas, (int(self.nodes[i, j][0]), int(self.nodes[i, j][1])), 5, (0, 0, 255), -1)
                    pass
                else:
                    if self.cells[(i - 1) // 2, (j - 1) // 2] == 1:
                        cv2.circle(self.canvas, (int(self.nodes[i, j][0]), int(self.nodes[i, j][1])), 5, (255, 0, 0),
                                   -1)
                    # cv2.circle(self.canvas, (int(self.nodes[i, j][0]), int(self.nodes[i, j][1])), 5, (255, 0, 0), -1)
                    pass

        cv2.imshow('canvas', self.canvas)

    def update(self):
        self.apply_forces()
        self.velocity += self.acceleration * 0.001
        self.nodes += self.velocity * 0.001
        self.t = time.time() - self.t0
        if self.t < 1.0:
            self.nodes[0][0] += 1 * np.array([-1, -1])
            self.nodes[0][-1] += 1 * np.array([-1, 1])
            self.nodes[-1][0] += 1 * np.array([1, -1])
            self.nodes[-1][-1] += 1 * np.array([1, 1])
        pass

    def get_total_energy(self):
        energy = 0
        for i in prange(2 * self.m):
            for j in prange(2 * self.n):
                if i < 2 * self.m - 1:
                    energy += np.linalg.norm(self.nodes[i, j] - self.nodes[i + 1, j])
                if j < 2 * self.n - 1:
                    energy += np.linalg.norm(self.nodes[i, j] - self.nodes[i, j + 1])
        return energy

    def spin(self):
        while True:
            # self.t = time.time() - self.t0
            # print(self.t)
            self.update()
            self.draw()
            # self.fps = 1 / (time.time() - self.t - self.t0)
            # print(self.fps)
            # show fps in upper left corner
            cv2.putText(self.canvas, f'fps: {self.fps:.2f}', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    bg = BioGrid(10, 10)
    bg.init_nodes()
    bg.draw()
    bg.spin()
    cv2.destroyAllWindows()
