import numpy as np
import vpython as vp


class LiveOrigamiFoldGeometry:
    def __init__(self, a, xn, yn, visualize=True):
        self.params = ['LiveOrigamiFoldGeometry', a, xn, yn]
        self.a = a
        self.b = a * 2 ** 0.5
        self.xn = xn
        self.yn = yn
        self.visualize = visualize
        self.r_str = 1.0
        self.pf_th = 0.0
        self.str_len = 1
        self.str_th = 0.3 * self.a

        self.i_max = 2 * self.xn + 3
        self.j_max = self.yn + 1

        self.i_fold = self.i_max // 2 - 1

        self.nodes = np.zeros((self.i_max, self.j_max, 3), dtype=np.float64)
        self.bars = []
        self.hinges = []
        self.vhinges = []
        self.hhinges = []

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

    def init_scene(self):
        self.scene = vp.canvas(title='Live Origami Fold Simulation', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(int(self.a * self.i_max / 2), int(self.a * self.j_max / 2), 0),
                               background=vp.color.white, fov=1)

    def update_nodes(self):
        self.fa = self.a * 2 ** 0.5
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        for i in range(0, self.i_max - 2):
            for j in range(0, self.j_max):
                self.nodes[i, j] = np.array([(i - self.i_fold) * self.a * np.cos(self.pf_th / 2),
                                             j * self.a,
                                             (i - self.i_fold) * self.a * np.sin(self.pf_th / 2) * np.sign(
                                                 i - self.i_fold)])

        for i in range(0, 2):
            for j in range(0, self.j_max):
                self.nodes[self.i_max - i - 1, j] = self.nodes[self.i_fold - self.str_len * (1 - 2 * i), j] + np.array(
                    [self.str_th * np.sin(self.pf_th / 2) * np.sign(0.5 - i), 0, self.str_th * np.cos(self.pf_th / 2)])

        if self.visualize:
            self.update_vp_nodes()

    def add_bar(self, p0, p1, l0, typ=None):
        self.bars.append([p0, p1, l0, typ])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, typ):
        # if type == 'hfold':
        #     self.hhinges.append([p0, p1, pa, pb, th0, l0, type])
        # elif type == 'vfold':
        #     self.vhinges.append([p0, p1, pa, pb, th0, l0, type])
        # else:
        #     self.hinges.append([p0, p1, pa, pb, th0, l0, type])

        self.hinges.append([p0, p1, pa, pb, th0, l0, typ])

    def update_vp_nodes(self, colors=None):
        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.vp_nodes[i][j].pos = vp.vector(self.nodes[i, j, 0], self.nodes[i, j, 1], self.nodes[i, j, 2])
                if colors is not None:
                    self.vp_nodes[i][j].color = colors[i][j]
        self.update_curves()

    def add_bar_hinges(self):
        # edges
        [self.add_bar([i, j], [i + 1, j], self.a) for i in range(self.i_max - 3) for j in range(self.j_max)]
        [self.add_bar([i, j], [i, j + 1], self.a) for j in range(self.j_max - 1) for i in range(self.i_max - 2)]

        # facets
        [self.add_bar([i, j], [i + 1, j + 1], self.b) for i in range(self.i_max - 3) for j in range(self.j_max - 1)]

        [self.add_hinge([i, j], [i + 1, j + 1], [i, j + 1], [i + 1, j], np.pi, self.b, 'facet') for i in
         range(self.i_max - 2 - 1) for j in range(self.j_max - 1)]

        [self.add_bar([i + 1, j], [i, j + 1], self.b) for i in range(self.i_max - 3) for j in range(self.j_max - 1)]

        [self.add_hinge([i + 1, j], [i, j + 1], [i, j], [i + 1, j + 1], np.pi, self.b, 'facet') for i in
         range(self.i_max - 2 - 1) for j in range(self.j_max - 1)]

        [self.add_hinge([i, j], [i + 1, j], [i + 1, j + 1], [i, j - 1], np.pi, self.a, 'facet') for i in
         range(self.i_max - 2 - 1) for j in range(1, self.j_max - 1)]

        # folds

        for i in range(1, self.i_max - 3):
            for j in range(self.j_max - 1):
                if i != self.i_fold:
                    tp = 'facet'
                    self.add_hinge([i, j], [i, j + 1], [i + 1, j], [i - 1, j + 1], np.pi, self.a, tp)
                else:
                    tp = 'fold'
                    self.add_hinge([i, j], [i, j + 1], [i + 1, j], [i - 1, j + 1], np.pi + self.str_th, self.a, tp)

        # hyrogels

        for i in range(0, 2):
            for j in range(1, self.j_max - 1):
                self.add_bar([self.i_max - i - 1, j], [self.i_fold - self.str_len * (1 - 2 * i), j], self.str_th,
                             'hydrogel_thickness')

        for j in range(1, self.j_max - 1):
            self.add_hinge([self.i_fold - self.str_len, j], [self.i_fold - self.str_len, j + 1],
                           [self.i_max - 1, j], [self.i_fold, j], np.pi / 2, self.a, 'facet')
            self.add_hinge([self.i_fold + self.str_len, j], [self.i_fold + self.str_len, j + 1],
                           [self.i_fold, j], [self.i_max - 2, j], np.pi / 2, self.a, 'facet')

            self.add_hinge([self.i_fold - self.str_len, j], [self.i_fold, j],
                           [self.i_max - 1, j], [self.i_fold - self.str_len, j - 1], np.pi / 2, self.a, 'facet')
            self.add_hinge([self.i_fold + self.str_len, j], [self.i_fold, j],
                           [self.i_fold + self.str_len, j - 1], [self.i_max - 2, j], np.pi / 2, self.a, 'facet')

        for j in range(1, self.j_max - 1):
            self.add_bar([self.i_max - 1, j], [self.i_max - 2, j],
                         2 * self.a * self.str_len * self.r_str * np.cos(self.pf_th / 2), 'hydrogel')

    def draw(self):
        if not self.visualize:
            return
        for j in range(self.j_max - 1):
            for i in range(self.i_max - 3):
                vp.triangle(vs=[self.vp_nodes[i][j],
                                self.vp_nodes[i + 1][j + 1],
                                self.vp_nodes[i][j + 1]],
                            color=vp.color.red)

                vp.triangle(vs=[self.vp_nodes[i][j],
                                self.vp_nodes[i + 1][j + 1],
                                self.vp_nodes[i + 1][j]],
                            color=vp.color.red)

                self.facet_curves.append(vp.curve(pos=[self.vp_nodes[i][j].pos,
                                                       self.vp_nodes[i + 1][j + 1].pos,
                                                       self.vp_nodes[i][j + 1].pos,
                                                       self.vp_nodes[i + 1][j].pos,
                                                       self.vp_nodes[i][j].pos],
                                                  color=vp.color.purple,
                                                  radius=0.01))

                self.crease_curves.append(vp.curve(pos=[self.vp_nodes[i][j].pos,
                                                        self.vp_nodes[i + 1][j].pos,
                                                        self.vp_nodes[i + 1][j + 1].pos,
                                                        self.vp_nodes[i][j + 1].pos,
                                                        self.vp_nodes[i][j].pos],
                                                   color=vp.color.black,
                                                   radius=0.01))

        for j in range(self.j_max):
            self.hydrogel_curves.append(vp.curve(pos=[self.vp_nodes[self.i_max - 1][j].pos,
                                                      self.vp_nodes[self.i_max - 2][j].pos],
                                                 color=vp.color.green,
                                                 radius=0.03))

        for i in range(0, 2):
            for j in range(1, self.j_max - 1):
                # self.add_bar([self.i_max - i - 1, j], [self.i_fold - self.str_len * (1 - 2 * i), j], self.str_th,
                #              'hydrogel_thickness')
                self.hydrogel_support_curves.append(vp.curve(pos=[self.vp_nodes[self.i_max - 1 - i][j].pos,
                                                                  self.vp_nodes[
                                                                      self.i_fold - self.str_len * (1 - 2 * i)][j].pos],
                                                             color=vp.color.green,
                                                             radius=0.03))

        # # draw spheres at the nodes
        # for i in range(self.i_max):
        #     for j in range(self.j_max):
        #         self.node_points.append(vp.sphere(pos=self.vp_nodes[i][j].pos, radius=0.05, color=vp.color.red))

    def update_curves(self):
        n = 0
        for i in range(0, self.i_max - 3):
            for j in range(0, self.j_max - 1):
                self.crease_curves[n].modify(0, pos=self.vp_nodes[i][j].pos)
                self.crease_curves[n].modify(1, pos=self.vp_nodes[i + 1][j].pos)
                self.crease_curves[n].modify(2, pos=self.vp_nodes[i + 1][j + 1].pos)
                self.crease_curves[n].modify(3, pos=self.vp_nodes[i][j + 1].pos)
                self.crease_curves[n].modify(4, pos=self.vp_nodes[i][j].pos)
                self.facet_curves[n].modify(0, pos=self.vp_nodes[i][j].pos)
                self.facet_curves[n].modify(1, pos=self.vp_nodes[i + 1][j + 1].pos)
                self.facet_curves[n].modify(2, pos=self.vp_nodes[i][j + 1].pos)
                self.facet_curves[n].modify(3, pos=self.vp_nodes[i + 1][j].pos)
                self.facet_curves[n].modify(4, pos=self.vp_nodes[i][j].pos)
                n += 1

        for j in range(1, self.j_max - 1):
            self.hydrogel_curves[j].modify(0, pos=self.vp_nodes[self.i_max - 1][j].pos)
            self.hydrogel_curves[j].modify(1, pos=self.vp_nodes[self.i_max - 2][j].pos)

        for i in range(0, 2):
            for j in range(1, self.j_max - 1):
                # self.add_bar([self.i_max - i - 1, j], [self.i_fold - self.str_len * (1 - 2 * i), j], self.str_th,
                #              'hydrogel_thickness')
                self.hydrogel_support_curves[i * (self.j_max - 3) + j].modify(0,
                                                                              pos=self.vp_nodes[self.i_max - 1 - i][
                                                                                  j].pos)
                self.hydrogel_support_curves[i * (self.j_max - 3) + j].modify(1,
                                                                              pos=self.vp_nodes[
                                                                                  self.i_fold - self.str_len * (
                                                                                          1 - 2 * i)][
                                                                                  j].pos)

        # # draw spheres at the nodes
        # for i in range(self.i_max):
        #     for j in range(self.j_max):
        #         self.node_points[i * self.j_max + j].pos = self.vp_nodes[i][j].pos

    def widget(self):
        vp.slider(min=0, max=2, value=self.a, bind=self.slider_theta)
        vp.slider(min=-np.pi / 2, max=np.pi / 2, value=self.pf_th, bind=self.slider_gamma)

    def slider_theta(self, a):
        self.a = a.value

    def slider_gamma(self, n):
        self.pf_th = n.value

    def spin(self):
        if self.visualize:
            self.widget()
            while True:
                self.update_nodes()
                vp.rate(30)


if __name__ == '__main__':
    a = 0.9
    n = 3
    xn = 3
    yn = 6

    geo = LiveOrigamiFoldGeometry(a, xn, yn, True)
    geo.spin()
