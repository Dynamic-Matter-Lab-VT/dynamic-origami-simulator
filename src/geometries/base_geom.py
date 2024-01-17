import numpy as np
import vpython as vp


class BaseGeometry:
    def __init__(self, a, b, xn, yn, visualize=True):
        self.params = ['RubberSheetGeometry', a, b, xn, yn]
        self.a = a
        self.b = b
        self.xn = xn
        self.yn = yn
        self.visualize = visualize
        self.fa = 1

        self.i_max = self.xn
        self.j_max = self.yn

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
        self.scene = vp.canvas(title='Rubber Sheet Geometry', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(int(self.a * self.i_max / 2), int(self.b * self.j_max / 2), 0),
                               background=vp.color.black, fov=1)

    def update_nodes(self):
        self.fa = (self.a ** 2 + self.b ** 2) ** 0.5
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.nodes[i, j] = np.array([i * self.a, j * self.b, 0])

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
        [self.add_bar([i, 0], [i + 1, 0], np.linalg.norm(self.nodes[i, 0] - self.nodes[i + 1, 0])) for i in
         range(0, self.i_max - 1)]
        [self.add_bar([i, self.j_max - 1], [i + 1, self.j_max - 1],
                      np.linalg.norm(self.nodes[i, self.j_max - 1] - self.nodes[i + 1, self.j_max - 1])) for i in
         range(0, self.i_max - 1)]
        [self.add_bar([0, j], [0, j + 1], np.linalg.norm(self.nodes[0, j] - self.nodes[0, j + 1])) for j in
         range(0, self.j_max - 1)]
        [self.add_bar([self.i_max - 1, j], [self.i_max - 1, j + 1],
                      np.linalg.norm(self.nodes[self.i_max - 1, j] - self.nodes[self.i_max - 1, j + 1])) for j in
         range(0, self.j_max - 1)]

        # facets
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                l0 = np.linalg.norm(self.nodes[i + 1, j + 1] - self.nodes[i, j])
                self.add_bar([i, j], [i + 1, j + 1], l0)
                self.add_hinge([i, j], [i + 1, j + 1],
                               [i, j + 1], [i + 1, j], np.pi, l0, 'facet')

        # folds
        for j in range(1, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                l0 = np.linalg.norm(self.nodes[i + 1, j] - self.nodes[i, j])
                self.add_bar([i, j], [i + 1, j], l0)
                self.add_hinge([i, j], [i + 1, j], [i + 1, j + 1], [i, j - 1], np.pi, l0, 'fold')

        for j in range(0, self.j_max - 1):
            for i in range(1, self.i_max - 1):
                l0 = np.linalg.norm(self.nodes[i, j + 1] - self.nodes[i, j])
                self.add_bar([i, j], [i, j + 1], l0)
                self.add_hinge([i, j], [i, j + 1], [i - 1, j], [i + 1, j + 1], np.pi, l0, 'fold')

    def draw(self):
        if not self.visualize:
            return
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                vp.triangle(vs=[self.vp_nodes[i][j],
                                self.vp_nodes[i + 1][j + 1],
                                self.vp_nodes[i + 1][j]],
                            color=vp.color.red)

                vp.triangle(vs=[self.vp_nodes[i][j],
                                self.vp_nodes[i + 1][j + 1],
                                self.vp_nodes[i][j + 1]],
                            color=vp.color.red)

                self.crease_curves.append(vp.curve(pos=[self.vp_nodes[i][j].pos,
                                                        self.vp_nodes[i + 1][j].pos,
                                                        self.vp_nodes[i + 1][j + 1].pos,
                                                        self.vp_nodes[i][j + 1].pos,
                                                        self.vp_nodes[i][j].pos],
                                                   color=vp.color.gray(0.5),
                                                   radius=0.01))

                self.facet_curves.append(vp.curve(pos=[self.vp_nodes[i][j].pos,
                                                       self.vp_nodes[i + 1][j + 1].pos],
                                                  color=vp.color.black, radius=0.01))

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
                n += 1

    def widget(self):
        vp.slider(min=0, max=2, value=self.a, bind=self.slider_theta)
        vp.slider(min=0, max=2, value=self.b, bind=self.slider_gamma)

    def slider_theta(self, a):
        self.a = a.value

    def slider_gamma(self, b):
        self.b = b.value

    def spin(self):
        if self.visualize:
            self.widget()
            while True:
                self.update_nodes()
                vp.rate(30)


if __name__ == '__main__':
    a = 0.9
    b = 0.5
    cut_th = 0.001
    xn = 4
    yn = 6

    geo = BaseGeometry(a, b, xn, yn, True)
    geo.spin()
