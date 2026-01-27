import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


class GeometryFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Rubber Sheet Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_a", Slider(
                label="Parameter a",
                min_val=0.1, max_val=2.0,
                initial_val=self.geometry.a,
                callback=self._update_a
            ))
            
            self.ui_manager.add_widget("slider_b", Slider(
                label="Parameter b",
                min_val=0.1, max_val=2.0,
                initial_val=self.geometry.b,
                callback=self._update_b
            ))
        
        if self.camera:
            self.camera.distance = 10.0

    def _update_a(self, value):
        self.geometry.a = float(value)

    def _update_b(self, value):
        self.geometry.b = float(value)

    def render(self, time, dt):
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


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

        self.update_nodes()
        self.add_bar_hinges()

    def update_nodes(self):
        self.fa = (self.a ** 2 + self.b ** 2) ** 0.5
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.nodes[i, j] = np.array([i * self.a, j * self.b, 0])

    def add_bar(self, p0, p1, l0):
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        self.hinges.append([p0, p1, pa, pb, th0, l0, type])

    def update_vp_nodes(self, colors=None):
        # Deprecated: vpython visualization removed
        pass

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
                self.add_hinge([i, j], [i, j + 1],
                               [i, j + 1], [i + 1, j], np.pi, l0, 'facet')

        # folds
        for j in range(1, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                l0 = np.linalg.norm(self.nodes[i + 1, j] - self.nodes[i, j])
                self.add_bar([i, j], [i, j + 1], l0)
                self.add_hinge([i, j], [i, j + 1], [i + 1, j + 1], [i, j - 1], np.pi, l0, 'fold')

        for j in range(0, self.j_max - 1):
            for i in range(1, self.i_max - 1):
                l0 = np.linalg.norm(self.nodes[i, j + 1] - self.nodes[i, j])
                self.add_bar([i, j], [i, j + 1], l0)
                self.add_hinge([i, j], [i, j + 1], [i - 1, j], [i + 1, j + 1], np.pi, l0, 'fold')

    def draw_piviz(self):
        if not self.visualize:
            return
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                p00 = tuple(self.nodes[i, j])
                p10 = tuple(self.nodes[i + 1, j])
                p01 = tuple(self.nodes[i, j + 1])
                p11 = tuple(self.nodes[i + 1, j + 1])

                pgfx.draw_triangle(v1=p00, v2=p11, v3=p10, color=Colors.RED)
                pgfx.draw_triangle(v1=p00, v2=p11, v3=p01, color=Colors.RED)

                pgfx.draw_path(points=[p00, p10, p11, p01, p00],
                               color=(0.5, 0.5, 0.5),
                               width=1.0)

                pgfx.draw_line(start=p00, end=p11, color=Colors.BLACK, width=1.0)

    def spin(self):
        if self.visualize:
            studio = PiVizStudio(scene_fx=GeometryFX(self))
            studio.run()


if __name__ == '__main__':
    a = 0.9
    b = 0.5
    cut_th = 0.001
    xn = 4
    yn = 6

    geo = BaseGeometry(a, b, xn, yn, True)
    geo.spin()
