import numpy as np
import vpython as vp


class SquareTwistGeometry:
    def __init__(self, l, phi, visualize=True):
        self.params = ['SquareTwistGeometry', l, phi]
        self.l = l
        self.phi = phi
        self.xn = 3
        self.yn = 3
        self.visualize = visualize

        self.i_max = self.xn + 1
        self.j_max = self.yn + 1

        self.h_idxs = [[[1, 1], [0, 2], [0, 1], [1, 2]],
                       [[1, 2], [2, 3], [1, 3], [2, 2]],
                       [[2, 2], [3, 1], [3, 2], [2, 1]],
                       [[2, 1], [1, 0], [2, 0], [1, 1]],
                       [[1, 1], [2, 2], [1, 2], [2, 1]],
                       [[1, 0], [0, 1], [0, 0], [1, 1]],
                       [[0, 2], [1, 3], [0, 3], [1, 2]],
                       [[2, 3], [3, 2], [3, 3], [2, 2]],
                       [[3, 1], [2, 0], [3, 0], [2, 1]]]

        self.nodes = np.zeros((self.i_max, self.j_max, 3), dtype=np.float64)
        self.bars = []
        self.hinges = []

        if self.visualize:
            self.scene = None
            self.crease_curves = []
            self.facet_curves = []
            self.init_scene()
            self.vp_nodes = [[vp.vertex(pos=vp.vector(0, 0, 0), color=vp.color.white)
                              for j in range(0, self.j_max)] for i in range(0, self.i_max)]
            self.draw()

        self.update_nodes()
        self.add_bar_hinges()

    def init_scene(self):
        self.scene = vp.canvas(title='Square Twist Simulation', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(0, 0, 0), background=vp.color.white, fov=1)

    def update_nodes(self):
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        self.nodes[0, 0] = np.array(
            [-1 - 2 * np.sin(self.phi) - 2 * np.cos(self.phi), 1 + 2 * np.sin(self.phi) - 2 * np.cos(self.phi), 0])
        self.nodes[1, 0] = np.array([-1 - 2 * np.cos(self.phi), 1 + 2 * np.sin(self.phi), 0])
        self.nodes[2, 0] = np.array([1 - 2 * np.cos(self.phi), 1 + 2 * np.sin(self.phi), 0])
        self.nodes[3, 0] = np.array(
            [1 + 2 * np.sin(self.phi) - 2 * np.cos(self.phi), 1 + 2 * np.sin(self.phi) + 2 * np.cos(self.phi), 0])

        self.nodes[0, 1] = np.array([-1 - 2 * np.sin(self.phi), 1 - 2 * np.cos(self.phi), 0])
        self.nodes[1, 1] = np.array([-1, 1, 0])
        self.nodes[2, 1] = np.array([1, 1, 0])
        self.nodes[3, 1] = np.array([1 + 2 * np.sin(self.phi), 1 + 2 * np.cos(self.phi), 0])

        self.nodes[0, 2] = np.array([-1 - 2 * np.sin(self.phi), -1 - 2 * np.cos(self.phi), 0])
        self.nodes[1, 2] = np.array([-1, -1, 0])
        self.nodes[2, 2] = np.array([1, -1, 0])
        self.nodes[3, 2] = np.array([1 + 2 * np.sin(self.phi), -1 + 2 * np.cos(self.phi), 0])

        self.nodes[0, 3] = np.array(
            [-1 - 2 * np.sin(self.phi) + 2 * np.cos(self.phi), -1 - 2 * np.sin(self.phi) - 2 * np.cos(self.phi), 0])
        self.nodes[1, 3] = np.array([-1 + 2 * np.cos(self.phi), -1 - 2 * np.sin(self.phi), 0])
        self.nodes[2, 3] = np.array([1 + 2 * np.cos(self.phi), -1 - 2 * np.sin(self.phi), 0])
        self.nodes[3, 3] = np.array(
            [1 + 2 * np.sin(self.phi) + 2 * np.cos(self.phi), -1 - 2 * np.sin(self.phi) + 2 * np.cos(self.phi),
             0])

        self.nodes = self.nodes * self.l / 2

        if self.visualize:
            self.update_vp_nodes()

    def add_bar(self, p0, p1, l0):
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        self.hinges.append([p0, p1, pa, pb, th0, l0, type])

    def update_vp_nodes(self, colors=None):
        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.vp_nodes[i][j].pos = vp.vector(self.nodes[i, j, 0], self.nodes[i, j, 1], self.nodes[i, j, 2])
                if colors is not None:
                    self.vp_nodes[i][j].color = colors[i][j]
        self.update_curves()

    def add_bar_hinges(self):

        # edges

        [self.add_bar([i, j], [i + 1, j], np.linalg.norm(self.nodes[i][j] - self.nodes[i + 1][j])) for i in
         range(self.i_max - 1) for j in range(self.j_max)]
        [self.add_bar([i, j], [i, j + 1], np.linalg.norm(self.nodes[i][j] - self.nodes[i][j + 1])) for i in
         range(self.i_max) for j in range(self.j_max - 1)]

        # facets
        # [1][1], [2][2]
        # [1][1], [0][2]
        # [1][2], [2][3]
        # [2][2], [3][1]
        # [2][1], [1][0]

        # [1][0], [0][1]
        # [0][2], [1][3]
        # [2][3], [3][2]
        # [3][1], [2][0]

        # kites
        self.add_bar([1, 1], [0, 2], np.linalg.norm(self.nodes[1][1] - self.nodes[0][2]))
        self.add_bar([1, 2], [2, 3], np.linalg.norm(self.nodes[1][2] - self.nodes[2][3]))
        self.add_bar([2, 2], [3, 1], np.linalg.norm(self.nodes[2][2] - self.nodes[3][1]))
        self.add_bar([2, 1], [1, 0], np.linalg.norm(self.nodes[2][1] - self.nodes[1][0]))
        # squares
        self.add_bar([1, 1], [2, 2], np.linalg.norm(self.nodes[1][1] - self.nodes[2][2]))
        self.add_bar([1, 0], [0, 1], np.linalg.norm(self.nodes[1][0] - self.nodes[0][1]))
        self.add_bar([0, 2], [1, 3], np.linalg.norm(self.nodes[0][2] - self.nodes[1][3]))
        self.add_bar([2, 3], [3, 2], np.linalg.norm(self.nodes[2][3] - self.nodes[3][2]))
        self.add_bar([3, 1], [2, 0], np.linalg.norm(self.nodes[3][1] - self.nodes[2][0]))

        # hinges
        # square faces
        sq_diag = np.linalg.norm(self.nodes[1][1] - self.nodes[2][2])
        self.add_hinge([1, 1], [2, 2], [1, 2], [2, 1], np.pi, sq_diag, 'facet')
        self.add_hinge([1, 0], [0, 1], [0, 0], [1, 1], np.pi, sq_diag, 'facet')
        self.add_hinge([0, 2], [1, 3], [0, 3], [1, 2], np.pi, sq_diag, 'facet')
        self.add_hinge([2, 3], [3, 2], [3, 3], [2, 2], np.pi, sq_diag, 'facet')
        self.add_hinge([3, 1], [2, 0], [3, 0], [2, 1], np.pi, sq_diag, 'facet')

        # kite faces
        kt_diag = np.linalg.norm(self.nodes[1][1] - self.nodes[0][2])
        self.add_hinge([1, 1], [0, 2], [0, 1], [1, 2], np.pi, kt_diag, 'facet')
        self.add_hinge([1, 2], [2, 3], [1, 3], [2, 2], np.pi, kt_diag, 'facet')
        self.add_hinge([2, 2], [3, 1], [3, 2], [2, 1], np.pi, kt_diag, 'facet')
        self.add_hinge([2, 1], [1, 0], [2, 0], [1, 1], np.pi, kt_diag, 'facet')

        # folds
        dth = np.pi / 6

        # self.add_hinge([0, 1], [1, 1], [1, 0], [1, 2], dth, self.l, 'fold')
        # self.add_hinge([1, 0], [1, 1], [0, 1], [2, 0], dth, self.l, 'fold')
        # self.add_hinge([0, 2], [1, 2], [1, 3], [0, 1], dth, self.l, 'fold')
        # self.add_hinge([1, 3], [1, 2], [0, 2], [2, 3], dth, self.l, 'fold')
        # self.add_hinge([2, 2], [2, 3], [1, 3], [3, 2], dth, self.l, 'fold')
        # self.add_hinge([2, 2], [3, 2], [2, 1], [2, 3], dth, self.l, 'fold')
        # self.add_hinge([2, 1], [3, 1], [3, 2], [2, 0], dth, self.l, 'fold')
        # self.add_hinge([2, 1], [2, 0], [3, 1], [1, 1], dth, self.l, 'fold')

        self.add_hinge([1, 1], [1, 2], [2, 2], [0, 1], dth, self.l, 'fold')
        self.add_hinge([1, 2], [2, 2], [1, 1], [1, 3], dth, self.l, 'fold')
        self.add_hinge([2, 1], [2, 2], [3, 2], [1, 1], dth, self.l, 'fold')
        self.add_hinge([2, 1], [1, 1], [2, 2], [2, 0], dth, self.l, 'fold')

    def draw(self):
        if not self.visualize:
            return

        # facets
        for idx in self.h_idxs:
            vp.triangle(vs=[self.vp_nodes[idx[0][0]][idx[0][1]],
                            self.vp_nodes[idx[1][0]][idx[1][1]],
                            self.vp_nodes[idx[2][0]][idx[2][1]]],
                        color=vp.color.red)
            vp.triangle(vs=[self.vp_nodes[idx[0][0]][idx[0][1]],
                            self.vp_nodes[idx[1][0]][idx[1][1]],
                            self.vp_nodes[idx[3][0]][idx[3][1]]],
                        color=vp.color.red)

            self.facet_curves.append(vp.curve(pos=[self.vp_nodes[idx[0][0]][idx[0][1]].pos,
                                                   self.vp_nodes[idx[1][0]][idx[1][1]].pos],
                                              color=vp.color.purple, radius=0.01))

        for j in range(self.j_max - 1):
            for i in range(self.i_max - 1):
                self.crease_curves.append(vp.curve(pos=[self.vp_nodes[i][j].pos,
                                                        self.vp_nodes[i + 1][j].pos,
                                                        self.vp_nodes[i + 1][j + 1].pos,
                                                        self.vp_nodes[i][j + 1].pos,
                                                        self.vp_nodes[i][j].pos],
                                                   color=vp.color.red,
                                                   radius=0.01))

    def update_curves(self):
        n = 0
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                self.crease_curves[n].modify(0, pos=self.vp_nodes[i][j].pos)
                self.crease_curves[n].modify(1, pos=self.vp_nodes[i + 1][j].pos)
                self.crease_curves[n].modify(2, pos=self.vp_nodes[i + 1][j + 1].pos)
                self.crease_curves[n].modify(3, pos=self.vp_nodes[i][j + 1].pos)
                self.crease_curves[n].modify(4, pos=self.vp_nodes[i][j].pos)
                n += 1

        n = 0
        for idx in self.h_idxs:
            self.facet_curves[n].modify(0, pos=self.vp_nodes[idx[0][0]][idx[0][1]].pos)
            self.facet_curves[n].modify(1, pos=self.vp_nodes[idx[1][0]][idx[1][1]].pos)
            n += 1

    def widget(self):
        vp.slider(min=0, max=np.pi / 2, value=self.phi, bind=self.slider_phi)

    def slider_phi(self, phi):
        self.phi = phi.value

    def spin(self):
        if self.visualize:
            self.widget()
            while True:
                self.update_nodes()
                vp.rate(30)


if __name__ == '__main__':
    geo = SquareTwistGeometry(1, np.pi / 6, True)
    geo.spin()
