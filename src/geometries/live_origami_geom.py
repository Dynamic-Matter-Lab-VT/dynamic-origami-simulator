import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


class LiveOrigamiFoldFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Live Origami Fold Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_a", Slider(
                label="Parameter a",
                min_val=0.1, max_val=2.0,
                initial_val=self.geometry.a,
                callback=self._update_a
            ))
            
            self.ui_manager.add_widget("slider_pf_th", Slider(
                label="pf_th",
                min_val=-np.pi / 2, max_val=np.pi / 2,
                initial_val=self.geometry.pf_th,
                callback=self._update_pf_th
            ))
        
        if self.camera:
            self.camera.distance = 15.0
            self.camera.set_view('iso')

    def _update_a(self, value):
        self.geometry.a = float(value)
        self.geometry.b = self.geometry.a * 2 ** 0.5

    def _update_pf_th(self, value):
        self.geometry.pf_th = float(value)

    def render(self, time, dt):
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


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

        self.update_nodes()
        self.add_bar_hinges()

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

    def add_bar(self, p0, p1, l0, typ=None):
        self.bars.append([p0, p1, l0, typ])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, typ):
        self.hinges.append([p0, p1, pa, pb, th0, l0, typ])

    def update_vp_nodes(self, colors=None):
        # Deprecated
        pass

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

    def draw_piviz(self):
        if not self.visualize:
            return
            
        for j in range(self.j_max - 1):
            for i in range(self.i_max - 3):
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

        # Hydrogel curves
        for j in range(self.j_max):
            start = tuple(self.nodes[self.i_max - 1, j])
            end = tuple(self.nodes[self.i_max - 2, j])
            pgfx.draw_line(start, end, color=Colors.GREEN, width=3.0)

        for i in range(0, 2):
            for j in range(1, self.j_max - 1):
                start = tuple(self.nodes[self.i_max - 1 - i, j])
                end = tuple(self.nodes[self.i_fold - self.str_len * (1 - 2 * i), j])
                pgfx.draw_line(start, end, color=Colors.GREEN, width=3.0)

    def spin(self):
        if self.visualize:
            studio = PiVizStudio(scene_fx=LiveOrigamiFoldFX(self))
            studio.run()


if __name__ == '__main__':
    a = 0.9
    n = 3
    xn = 3
    yn = 6

    geo = LiveOrigamiFoldGeometry(a, xn, yn, True)
    geo.spin()
