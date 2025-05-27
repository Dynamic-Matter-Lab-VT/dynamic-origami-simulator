import numpy as np
import vpython as vp


class BistableMaterialGeometry:
    def __init__(self, a, xn, yn, visualize=True):
        self.params = ['BistableMaterialGeometry', a, xn, yn]
        self.a = a
        self.b = a * 2 ** 0.5
        self.xn = xn
        self.yn = yn
        self.visualize = visualize
        self.r_str = 0.7

        self.i_max = self.xn + 1
        self.j_max = self.yn + 1

        self.nodes = np.zeros((self.i_max, self.j_max, 3), dtype=np.float64)
        self.bars = []
        self.hinges = []
        self.vhinges = []
        self.hhinges = []

        if self.visualize:
            self.scene = None
            self.crease_curves = []
            self.facet_curves = []
            self.stress_curves = []
            self.init_scene()
            self.vp_nodes = [[vp.vertex(pos=vp.vector(0, 0, 0), color=vp.color.white)
                              for j in range(0, self.j_max)] for i in range(0, self.i_max)]
            self.draw()

        self.update_nodes()
        self.add_bar_hinges()

    def init_scene(self):
        self.scene = vp.canvas(title='Bistable Material Simulation', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(int(self.a * self.i_max / 2), int(self.a * self.j_max / 2), 0),
                               background=vp.color.white, fov=1)

    def update_nodes(self):
        self.fa = self.a * 2 ** 0.5
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.nodes[i, j] = np.array([i * self.a, j * self.a, 0])

        if self.visualize:
            self.update_vp_nodes()

    def add_bar(self, p0, p1, l0):
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        if type == 'hfold':
            self.hhinges.append([p0, p1, pa, pb, th0, l0, type])
        elif type == 'vfold':
            self.vhinges.append([p0, p1, pa, pb, th0, l0, type])
        else:
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
        [self.add_bar([i, j], [i + 1, j], self.a) for i in range(self.i_max - 1) for j in range(self.j_max)]
        [self.add_bar([i, j], [i, j + 1], self.a) for j in range(self.j_max - 1) for i in range(self.i_max)]

        # facets
        [self.add_bar([i, j], [i + 1, j + 1], self.b) for i in range(self.i_max - 1) for j in range(self.j_max - 1)]

        [self.add_hinge([i, j], [i + 1, j + 1], [i, j + 1], [i + 1, j], np.pi, self.b, 'facet') for i in
         range(self.i_max - 1) for j in range(self.j_max - 1)]

        [self.add_bar([i + 1, j], [i, j + 1], self.b) for i in range(self.i_max - 1) for j in range(self.j_max - 1)]

        [self.add_hinge([i + 1, j], [i, j + 1], [i, j], [i + 1, j + 1], np.pi, self.b, 'facet') for i in
         range(self.i_max - 1) for j in range(self.j_max - 1)]

        # # folds
        [self.add_hinge([i, j], [i + 1, j], [i + 1, j + 1], [i, j - 1], np.pi, self.a, 'hfold') for i in
         range(self.i_max - 1) for j in range(1, self.j_max - 1)]

        [self.add_hinge([i, j], [i, j + 1], [i + 1, j], [i - 1, j + 1], np.pi, self.a, 'vfold') for j in
         range(self.j_max - 1) for i in range(1, self.i_max - 1)]

        # stress points
        for j in range(self.j_max):
            self.add_bar([0, j], [self.i_max - 1, j], self.a * self.i_max * self.r_str)

        for i in range(self.i_max):
            self.add_bar([i, 0], [i, self.j_max - 1], self.a * self.j_max * self.r_str)

    def draw(self):
        if not self.visualize:
            return
        for j in range(self.j_max - 1):
            for i in range(self.i_max - 1):
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
            self.stress_curves.append(vp.curve(pos=[self.vp_nodes[0][j].pos,
                                                    self.vp_nodes[self.i_max - 1][j].pos],
                                               color=vp.color.green,
                                               radius=0.01))
        for i in range(self.i_max):
            self.stress_curves.append(vp.curve(pos=[self.vp_nodes[i][0].pos,
                                                    self.vp_nodes[i][self.j_max - 1].pos],
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
                self.facet_curves[n].modify(0, pos=self.vp_nodes[i][j].pos)
                self.facet_curves[n].modify(1, pos=self.vp_nodes[i + 1][j + 1].pos)
                self.facet_curves[n].modify(2, pos=self.vp_nodes[i][j + 1].pos)
                self.facet_curves[n].modify(3, pos=self.vp_nodes[i + 1][j].pos)
                self.facet_curves[n].modify(4, pos=self.vp_nodes[i][j].pos)
                n += 1

        # update stress curves
        for i in range(0, self.i_max):
            self.stress_curves[i].modify(0, pos=self.vp_nodes[i][0].pos)
            self.stress_curves[i].modify(1, pos=self.vp_nodes[i][self.j_max - 1].pos)
        for j in range(0, self.j_max):
            self.stress_curves[self.i_max + j].modify(0, pos=self.vp_nodes[0][j].pos)
            self.stress_curves[self.i_max + j].modify(1, pos=self.vp_nodes[self.i_max - 1][j].pos)

    def widget(self):
        vp.slider(min=0, max=2, value=self.a, bind=self.slider_theta)
        vp.slider(min=0, max=2, value=self.b, bind=self.slider_gamma)

    def slider_theta(self, a):
        self.a = a.value

    def slider_gamma(self, n):
        self.n = int(n.value)

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
    yn = 10

    geo = BistableMaterialGeometry(a, xn, yn, True)
    geo.spin()
