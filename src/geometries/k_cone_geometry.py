import numpy as np
import vpython as vp


def get_rotation_matrix(angle, axis):
    """
    Get the rotation matrix for a given angle and axis.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    l = np.linalg.norm(axis)
    if l == 0:
        return np.eye(3)
    else:
        axis = axis / l
        x, y, z = axis
        return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                         [t * y * x + s * z, t * y * y + c, t * y * z - s * x],
                         [t * z * x - s * y, t * z * y + s * x, t * z * z + c]])


class KConeGeometry:
    def __init__(self, k=4, l=1.0, h=0.0, th_c=np.pi / 6, hg_th=0.1, visualize=True):
        self.params = ['KConeGeometry', k, l, h, th_c, hg_th]

        self.n = k  # number of sides
        self.l = l  # length of the side base
        self.h = h  # height of the hole
        self.th_c = th_c  # angle of the cut size
        self.hg_th = hg_th  # thickness of the hydrogel

        self.bounds = [0, 1, 5, 4]
        self.r = self.l / (2 * np.sin(np.pi / self.n))
        self.th_s = 2 * np.pi / self.n - self.th_c
        self.b = self.h / (np.cos(self.th_s / 2))
        self.a = self.l / (2 * np.sin(self.th_s / 2))
        self.phi = -np.arccos(np.tan(self.th_s / 2) / np.tan(np.pi / self.n))

        self.visualize = visualize

        self.i_max = 4 * self.n
        self.j_max = 1

        self.nodes = np.zeros((self.i_max, self.j_max, 3), dtype=np.float64)
        self.get_k_cone_coords()
        self.bars = []
        self.hinges = []

        if self.visualize:
            self.scene = None
            self.crease_curves = []
            self.facet_curves = []
            self.hydrogel_curves = []
            self.hydrogel_support_curves = []
            self.node_points = []
            self.init_scene()
            self.vp_nodes = [[vp.vertex(pos=vp.vector(0, 0, 0), color=vp.color.white)
                              for j in range(0, self.j_max)] for i in range(0, self.i_max)]
            self.draw()

        self.update_nodes()
        self.add_bar_hinges()

    def get_k_cone_coords(self):

        self.th_s = 2 * np.pi / self.n - np.abs(self.th_c)
        self.b = self.h / (np.cos(self.th_s / 2))
        self.a = self.l / (2 * np.sin(self.th_s / 2))

        x_s = np.zeros((6, 3))
        x_s[0, :] = self.b * np.array([-np.sin(self.th_s / 2), np.cos(self.th_s / 2), 0])
        x_s[1, :] = self.a * np.array([-np.sin(self.th_s / 2), np.cos(self.th_s / 2), 0])

        x_s[4, :] = self.b * np.array([np.sin(self.th_s / 2), np.cos(self.th_s / 2), 0])
        x_s[5, :] = self.a * np.array([np.sin(self.th_s / 2), np.cos(self.th_s / 2), 0])

        x_s[2, :] = np.sum(x_s[[0, 1, 4, 5], :], axis=0) / 4
        x_s[3, :] = np.sum(x_s[[0, 1, 4, 5], :], axis=0) / 4 + np.array([0, 0, self.hg_th])

        x_s = x_s - np.array([0, self.a * np.cos(self.th_s / 2), 0])

        self.phi = -np.arccos(np.tan(self.th_s / 2) / np.tan(np.pi / self.n)) * np.sign(self.th_c)

        for i in range(self.n):
            T1 = get_rotation_matrix(self.phi, [1, 0, 0])
            T2 = get_rotation_matrix(i * 2 * np.pi / self.n, [0, 0, 1])
            x = (T1 @ x_s.T).T + self.r * np.cos(np.pi / self.n) * np.array([0, 1, 0])
            x = (T2 @ x.T).T
            self.nodes[4 * i:4 * i + 4, 0, :] = x[:4, :]

    def init_scene(self):
        self.scene = vp.canvas(title='K-Cone Simulation', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(0, 0, 0),
                               background=vp.color.white, fov=1)

    def update_nodes(self):
        self.nodes = np.zeros((self.i_max, self.j_max, 3))
        self.get_k_cone_coords()

        if self.visualize:
            self.update_vp_nodes()

    def add_bar(self, p0, p1, l0=None, typ=None):
        if l0 is None:
            l0 = np.linalg.norm(self.nodes[p0[0]][p0[1]] - self.nodes[p1[0]][p1[1]])
        self.bars.append([p0, p1, l0, typ])

    def add_hinge(self, p0, p1, pa, pb, th0, l0=None, typ=None):
        if l0 is None:
            l0 = np.linalg.norm(self.nodes[p0[0]][p0[1]] - self.nodes[p1[0]][p1[1]])
        self.hinges.append([p0, p1, pa, pb, th0, l0, typ])

    def update_vp_nodes(self, colors=None):
        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.vp_nodes[i][j].pos = vp.vector(self.nodes[i, j, 0], self.nodes[i, j, 1], self.nodes[i, j, 2])
                if colors is not None:
                    self.vp_nodes[i][j].color = colors[i][j]
        self.update_curves()

    def add_bar_hinges(self):
        for k in range(self.n):
            for i in range(4):
                p0 = (self.bounds[i] + 4 * k) % (4 * self.n)
                p1 = (self.bounds[i - 1] + 4 * k) % (4 * self.n)
                p1_ = (self.bounds[(i + 1) % 4] + 4 * k) % (4 * self.n)
                p2 = (2 + 4 * (k + 1)) % (4 * self.n)
                p3 = (p2 + 1) % (4 * self.n)
                p4 = (3 + 4 * k) % (4 * self.n)

                self.add_bar([p0, 0], [p2, 0])
                self.add_bar([p0, 0], [p1, 0])

                self.add_hinge([p0, 0], [p2, 0], [p1, 0], [p1_, 0], np.pi, None, 'facet')

                self.add_bar([p2, 0], [p3, 0])
                self.add_bar([p3, 0], [p4, 0], None, 'hydrogel')

                self.add_hinge([p0, 0], [p2, 0], [p1, 0], [p3, 0], np.pi / 2, None, 'facet')

    def draw(self):
        if not self.visualize:
            return

        for k in range(self.n):
            self.node_points.append(
                vp.sphere(pos=vp.vector(self.nodes[(4 * k + 1) % (4 * self.n), 0, 0],
                                        self.nodes[(4 * k + 1) % (4 * self.n), 0, 1],
                                        self.nodes[(4 * k + 1) % (4 * self.n), 0, 2]),
                          radius=0.02, color=vp.color.black))
            for i in range(4):
                p0 = (self.bounds[i] + 4 * k) % (4 * self.n)
                p1 = (self.bounds[i - 1] + 4 * k) % (4 * self.n)
                p2 = (2 + 4 * (k + 1)) % (4 * self.n)
                p3 = (p2 + 1) % (4 * self.n)
                p4 = (3 + 4 * k) % (4 * self.n)

                vp.triangle(vs=[self.vp_nodes[p0][0],
                                self.vp_nodes[p1][0],
                                self.vp_nodes[p2][0]],
                            color=vp.color.red)

                self.crease_curves.append(vp.curve(pos=[self.vp_nodes[p0][0].pos, self.vp_nodes[p2][0].pos],
                                                   color=vp.color.blue, radius=0.005))
                self.facet_curves.append(vp.curve(pos=[self.vp_nodes[p0][0].pos, self.vp_nodes[p1][0].pos],
                                                  color=vp.color.red, radius=0.005))
                self.hydrogel_support_curves.append(vp.curve(pos=[self.vp_nodes[p2][0].pos, self.vp_nodes[p3][0].pos],
                                                             color=vp.color.green, radius=0.02))
                self.hydrogel_curves.append(vp.curve(pos=[self.vp_nodes[p4][0].pos, self.vp_nodes[p3][0].pos],
                                                     color=vp.color.green, radius=0.02))

    def update_curves(self):

        for k in range(self.n):
            self.node_points[k].pos = vp.vector(self.nodes[(4 * k + 1) % (4 * self.n), 0, 0],
                                                self.nodes[(4 * k + 1) % (4 * self.n), 0, 1],
                                                self.nodes[(4 * k + 1) % (4 * self.n), 0, 2])
            for i in range(4):
                p0 = (self.bounds[i] + 4 * k) % (4 * self.n)
                p1 = (self.bounds[i - 1] + 4 * k) % (4 * self.n)
                p2 = (2 + 4 * (k + 1)) % (4 * self.n)
                p3 = (p2 + 1) % (4 * self.n)
                p4 = (3 + 4 * k) % (4 * self.n)
                idx = 4 * k + i
                self.crease_curves[idx].modify(0, pos=self.vp_nodes[p0][0].pos)
                self.crease_curves[idx].modify(1, pos=self.vp_nodes[p2][0].pos)
                self.facet_curves[idx].modify(0, pos=self.vp_nodes[p0][0].pos)
                self.facet_curves[idx].modify(1, pos=self.vp_nodes[p1][0].pos)
                self.hydrogel_support_curves[idx].modify(0, pos=self.vp_nodes[p2][0].pos)
                self.hydrogel_support_curves[idx].modify(1, pos=self.vp_nodes[p3][0].pos)
                self.hydrogel_curves[idx].modify(0, pos=self.vp_nodes[p4][0].pos)
                self.hydrogel_curves[idx].modify(1, pos=self.vp_nodes[p3][0].pos)

    def widget(self):
        self.scene.append_to_caption('\n th_c: ')
        vp.slider(min=-np.pi / 4, max=np.pi / 4, value=self.th_c, bind=self.slider_th_c)
        self.scene.append_to_caption('\n hg_th: ')
        vp.slider(min=0, max=self.l, value=self.hg_th, bind=self.slider_hg_th)
        self.scene.append_to_caption('\n h: ')
        vp.slider(min=0, max=self.l, value=self.h, bind=self.slider_h)

    def slider_th_c(self, th):
        self.th_c = th.value

    def slider_hg_th(self, th):
        self.hg_th = th.value

    def slider_h(self, th):
        self.h = th.value

    def spin(self):
        if self.visualize:
            self.widget()
            while True:
                self.update_nodes()
                vp.rate(30)


if __name__ == '__main__':
    # a = 0.9
    # n = 3
    # xn = 3
    # yn = 6

    geo = KConeGeometry(5)
    geo.spin()
