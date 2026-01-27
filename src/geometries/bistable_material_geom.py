import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


class BistableMaterialFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Bistable Material Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_a", Slider(
                label="Parameter a",
                min_val=0.1, max_val=2.0,
                initial_val=self.geometry.a,
                callback=self._update_a
            ))
            
            self.ui_manager.add_widget("slider_r_str", Slider(
                label="r_str",
                min_val=0.1, max_val=2.0,
                initial_val=self.geometry.r_str,
                callback=self._update_r_str
            ))
        
        if self.camera:
            self.camera.distance = 15.0
            self.camera.set_view('iso')

    def _update_a(self, value):
        self.geometry.a = float(value)
        self.geometry.b = self.geometry.a * 2 ** 0.5

    def _update_r_str(self, value):
        self.geometry.r_str = float(value)

    def render(self, time, dt):
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


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

        self.update_nodes()
        self.add_bar_hinges()

    def update_nodes(self):
        self.fa = self.a * 2 ** 0.5
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.nodes[i, j] = np.array([i * self.a, j * self.a, 0])

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
        # Deprecated
        pass

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

    def draw_piviz(self):
        if not self.visualize:
            return
            
        for j in range(self.j_max - 1):
            for i in range(self.i_max - 1):
                v00 = tuple(self.nodes[i, j])
                v10 = tuple(self.nodes[i + 1, j])
                v01 = tuple(self.nodes[i, j + 1])
                v11 = tuple(self.nodes[i + 1, j + 1])
                
                # Triangles
                pgfx.draw_triangle(v00, v11, v01, color=Colors.RED)
                pgfx.draw_triangle(v00, v11, v10, color=Colors.RED)
                
                # Facet curves (Purple)
                pgfx.draw_path([v00, v11, v01, v10, v00], color=Colors.PURPLE, width=1.0)
                
                # Crease curves (Black)
                pgfx.draw_path([v00, v10, v11, v01, v00], color=Colors.BLACK, width=1.0)

        # Stress curves
        for j in range(self.j_max):
            start = tuple(self.nodes[0, j])
            end = tuple(self.nodes[self.i_max - 1, j])
            pgfx.draw_line(start, end, color=Colors.GREEN, width=1.0)
            
        for i in range(self.i_max):
            start = tuple(self.nodes[i, 0])
            end = tuple(self.nodes[i, self.j_max - 1])
            pgfx.draw_line(start, end, color=Colors.RED, width=1.0)

    def spin(self):
        if self.visualize:
            studio = PiVizStudio(scene_fx=BistableMaterialFX(self))
            studio.run()


if __name__ == '__main__':
    a = 0.9
    n = 3
    xn = 3
    yn = 10

    geo = BistableMaterialGeometry(a, xn, yn, True)
    geo.spin()
