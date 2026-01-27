import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


class SquareTwistFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Square Twist Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_phi", Slider(
                label="Phi",
                min_val=0.0, max_val=np.pi / 2,
                initial_val=self.geometry.phi,
                callback=self._update_phi
            ))
        
        if self.camera:
            self.camera.distance = 10.0
            self.camera.set_view('iso')

    def _update_phi(self, value):
        self.geometry.phi = float(value)

    def render(self, time, dt):
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


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

        self.update_nodes()
        self.add_bar_hinges()

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

    def add_bar(self, p0, p1, l0):
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        self.hinges.append([p0, p1, pa, pb, th0, l0, type])

    def update_vp_nodes(self, colors=None):
        # Deprecated
        pass

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

    def draw_piviz(self):
        if not self.visualize:
            return

        # facets
        for idx in self.h_idxs:
            v0 = tuple(self.nodes[idx[0][0]][idx[0][1]])
            v1 = tuple(self.nodes[idx[1][0]][idx[1][1]])
            v2 = tuple(self.nodes[idx[2][0]][idx[2][1]])
            v3 = tuple(self.nodes[idx[3][0]][idx[3][1]])
            
            pgfx.draw_triangle(v0, v1, v2, color=Colors.RED)
            pgfx.draw_triangle(v0, v1, v3, color=Colors.RED)

            # Facet diagonal
            pgfx.draw_line(v0, v1, color=Colors.PURPLE, width=1.0)

        # Crease curves
        for j in range(self.j_max - 1):
            for i in range(self.i_max - 1):
                points = [
                    tuple(self.nodes[i, j]),
                    tuple(self.nodes[i + 1, j]),
                    tuple(self.nodes[i + 1, j + 1]),
                    tuple(self.nodes[i, j + 1]),
                    tuple(self.nodes[i, j])
                ]
                pgfx.draw_path(points, color=Colors.RED, width=1.0)

    def spin(self):
        if self.visualize:
            studio = PiVizStudio(scene_fx=SquareTwistFX(self))
            studio.run()


if __name__ == '__main__':
    geo = SquareTwistGeometry(1, np.pi / 6, True)
    geo.spin()
