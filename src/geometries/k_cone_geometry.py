import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


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


class KConeFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("K-Cone Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_th_c", Slider(
                label="th_c",
                min_val=-np.pi / 4, max_val=np.pi / 4,
                initial_val=self.geometry.th_c,
                callback=self._update_th_c
            ))
            
            self.ui_manager.add_widget("slider_hg_th", Slider(
                label="hg_th",
                min_val=0.0, max_val=self.geometry.l,
                initial_val=self.geometry.hg_th,
                callback=self._update_hg_th
            ))
            
            self.ui_manager.add_widget("slider_h", Slider(
                label="h",
                min_val=0.0, max_val=self.geometry.l,
                initial_val=self.geometry.h,
                callback=self._update_h
            ))
        
        if self.camera:
            self.camera.distance = 5.0
            self.camera.set_view('iso')

    def _update_th_c(self, value):
        self.geometry.th_c = float(value)

    def _update_hg_th(self, value):
        self.geometry.hg_th = float(value)

    def _update_h(self, value):
        self.geometry.h = float(value)

    def render(self, time, dt):
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


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

    def update_nodes(self):
        self.nodes = np.zeros((self.i_max, self.j_max, 3))
        self.get_k_cone_coords()

    def add_bar(self, p0, p1, l0=None, typ=None):
        if l0 is None:
            l0 = np.linalg.norm(self.nodes[p0[0]][p0[1]] - self.nodes[p1[0]][p1[1]])
        self.bars.append([p0, p1, l0, typ])

    def add_hinge(self, p0, p1, pa, pb, th0, l0=None, typ=None):
        if l0 is None:
            l0 = np.linalg.norm(self.nodes[p0[0]][p0[1]] - self.nodes[p1[0]][p1[1]])
        self.hinges.append([p0, p1, pa, pb, th0, l0, typ])

    def update_vp_nodes(self, colors=None):
        # Deprecated
        pass

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

    def draw_piviz(self):
        if not self.visualize:
            return

        for k in range(self.n):
            # Node sphere
            node_idx = (4 * k + 1) % (4 * self.n)
            pos = tuple(self.nodes[node_idx, 0])
            pgfx.draw_sphere(center=pos, radius=0.02, color=Colors.BLACK)
            
            for i in range(4):
                p0 = (self.bounds[i] + 4 * k) % (4 * self.n)
                p1 = (self.bounds[i - 1] + 4 * k) % (4 * self.n)
                p2 = (2 + 4 * (k + 1)) % (4 * self.n)
                p3 = (p2 + 1) % (4 * self.n)
                p4 = (3 + 4 * k) % (4 * self.n)

                v0 = tuple(self.nodes[p0, 0])
                v1 = tuple(self.nodes[p1, 0])
                v2 = tuple(self.nodes[p2, 0])
                v3 = tuple(self.nodes[p3, 0])
                v4 = tuple(self.nodes[p4, 0])

                # Triangle
                pgfx.draw_triangle(v0, v1, v2, color=Colors.RED)

                # Crease curve (Blue)
                pgfx.draw_line(v0, v2, color=Colors.BLUE, width=2.0)
                
                # Facet curve (Red)
                pgfx.draw_line(v0, v1, color=Colors.RED, width=2.0)
                
                # Hydrogel support (Green)
                pgfx.draw_line(v2, v3, color=Colors.GREEN, width=4.0)
                
                # Hydrogel (Green)
                pgfx.draw_line(v4, v3, color=Colors.GREEN, width=4.0)

    def spin(self):
        if self.visualize:
            studio = PiVizStudio(scene_fx=KConeFX(self))
            studio.run()


if __name__ == '__main__':
    # a = 0.9
    # n = 3
    # xn = 3
    # yn = 6

    geo = KConeGeometry(5)
    geo.spin()
