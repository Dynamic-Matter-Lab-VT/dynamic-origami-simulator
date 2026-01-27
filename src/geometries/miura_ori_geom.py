"""
MiuraOriGeometry Class

This class represents the geometry of a Miura-Ori origami structure. It provides methods to initialize the geometry,
update dimensions, update nodes, add bars and hinges, and visualize the geometry using PiViz.

Author: Yogesh Phalak
Date: 2023-06-20
Filename: miura_ori_geom.py

"""

import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


class MiuraOriFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Miura-Ori Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_theta", Slider(
                label="Theta",
                min_val=0.1, max_val=np.pi / 2 - 0.1,
                initial_val=self.geometry.theta,
                callback=self._update_theta
            ))
            
            self.ui_manager.add_widget("slider_gamma", Slider(
                label="Gamma",
                min_val=0.1, max_val=np.pi / 2 - 0.1,
                initial_val=self.geometry.gamma,
                callback=self._update_gamma
            ))
        
        if self.camera:
            self.camera.distance = 15.0
            self.camera.set_view('iso')

    def _update_theta(self, value):
        self.geometry.theta = float(value)

    def _update_gamma(self, value):
        self.geometry.gamma = float(value)

    def render(self, time, dt):
        self.geometry.update_dimensions()
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


class MiuraOriGeometry:
    def __init__(self, a, b, gamma, theta, xn, yn, visualize=True):
        """Initialize the MiuraOriGeometry instance with specified parameters."""
        self.params = ['MiuraOri', a, b, gamma, theta, xn, yn]
        self.a = a
        self.b = b
        self.gamma = gamma
        self.theta = theta
        self.xn = xn
        self.yn = yn
        self.visualize = visualize

        self.i_max = 2 * self.xn + 1
        self.j_max = 2 * self.yn + 1

        self.ht = 0
        self.t = 0
        self.l = 0
        self.w = 0
        self.v = 0
        self.fa = 0

        self.nodes = np.zeros((self.i_max, self.j_max, 3), dtype=np.float64)
        self.bars = []
        self.hinges = []
        self.update_dimensions()

        self.update_nodes()
        self.add_bar_hinges()

    def update_dimensions(self):
        """Update the dimensions of the Miura-Ori geometry."""
        self.ht = self.a * np.sin(self.gamma) * np.sin(self.theta)
        self.t = self.a * np.sin(self.gamma) * np.sin(self.theta)
        self.l = self.b * np.tan(self.gamma) * np.cos(self.theta) / (
                (1 + np.cos(self.theta) ** 2 * np.tan(self.gamma) ** 2) ** 0.5)
        self.w = self.a * (1 - np.sin(self.theta) ** 2 * np.sin(self.gamma) ** 2) ** 0.5
        self.v = self.b / (1 + np.cos(self.theta) ** 2 * np.tan(self.gamma) ** 2) ** 0.5
        self.fa = (self.a ** 2 + self.b ** 2 - 2 * self.a * self.b * np.cos(np.pi - self.gamma)) ** 0.5

    def update_nodes(self):
        """Update the positions of the nodes in the Miura-Ori geometry."""
        self.nodes = np.zeros((self.i_max, self.j_max, 3))
        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.nodes[i, j] = np.array([i * self.l, j * self.w + self.v * (i % 2), self.ht * (j % 2)])

    def add_bar(self, p0, p1, l0):
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        self.hinges.append([p0, p1, pa, pb, th0, l0, type])

    def update_vp_nodes(self, colors=None):
        # Deprecated: vpython visualization removed
        pass

    def add_bar_hinges(self):
        """Add bars and hinges to the geometry based on the current parameters."""
        th_h = 2 * self.theta
        th_v = np.arccos(-(np.sin(self.gamma) ** 2 * np.sin(self.theta) ** 2 - 2 * np.sin(self.theta) ** 2 + 1)
                         / (np.sin(self.gamma) ** 2 * np.sin(self.theta) ** 2 - 1))

        # edges
        [self.add_bar([i, 0], [i + 1, 0], self.b) for i in range(0, self.i_max - 1)]
        [self.add_bar([i, self.j_max - 1], [i + 1, self.j_max - 1], self.b) for i in range(0, self.i_max - 1)]
        [self.add_bar([0, j], [0, j + 1], self.a) for j in range(0, self.j_max - 1)]
        [self.add_bar([self.i_max - 1, j], [self.i_max - 1, j + 1], self.a) for j in range(0, self.j_max - 1)]

        # facets
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                self.add_bar([i + i % 2, j], [i + 1 - i % 2, j + 1], self.fa)
                self.add_hinge([i + i % 2, j], [i + 1 - i % 2, j + 1],
                               [i, j + 1 - i % 2], [i + 1, j + i % 2], np.pi, self.fa, 'facet')

        # folds
        for j in range(1, self.j_max - 1):
            [self.add_bar([i, j], [i + 1, j], self.b) for i in range(0, self.i_max - 1)]
            [self.add_hinge([i, j], [i + 1, j],
                            [i + 1 - i % 2, j + 1], [i + i % 2, j - 1],
                            np.pi + th_h * (-1 + 2 * (j % 2)), self.b, 'fold') for i in range(0, self.i_max - 1)]

        for i in range(1, self.i_max - 1):
            [self.add_bar([i, j], [i, j + 1], self.a) for j in range(0, self.j_max - 1)]
            [self.add_hinge([i, j], [i, j + 1],
                            [i - 1, j + 1 - i % 2], [i + 1, j + 1 - i % 2],
                            np.pi + th_v * (1 - 2 * (j % 2)) * (1 - 2 * (i % 2)), self.a, 'fold') for j in
             range(0, self.j_max - 1)]

    def draw_piviz(self):
        """Draw the geometry using PiViz."""
        if not self.visualize:
            return
            
        # Draw triangles (facets)
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                # The original code drew 4 triangles per cell (p=0..1, q=0..1)
                # [vp.triangle(vs=[self.vp_nodes[i][j + i % 2],
                #                  self.vp_nodes[i + 1][j + 1 - i % 2],
                #                  self.vp_nodes[i + p][j + q]],
                #              color=vp.color.red) for p in range(0, 2) for q in range(0, 2)]
                
                # Let's replicate this logic
                v1 = tuple(self.nodes[i, j + i % 2])
                v2 = tuple(self.nodes[i + 1, j + 1 - i % 2])
                
                for p in range(0, 2):
                    for q in range(0, 2):
                        v3 = tuple(self.nodes[i + p, j + q])
                        pgfx.draw_triangle(v1, v2, v3, color=Colors.RED)
                        
        # Draw crease curves (outlines)
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                points = [
                    tuple(self.nodes[i, j]),
                    tuple(self.nodes[i + 1, j]),
                    tuple(self.nodes[i + 1, j + 1]),
                    tuple(self.nodes[i, j + 1]),
                    tuple(self.nodes[i, j])
                ]
                pgfx.draw_path(points, color=Colors.BLACK, width=1.0)
                
                # Facet diagonal
                p_start = tuple(self.nodes[i, j + i % 2])
                p_end = tuple(self.nodes[i + 1, j + 1 - i % 2])
                pgfx.draw_line(p_start, p_end, color=Colors.BLACK, width=1.0)

    def spin(self):
        """Starts the interactive animation of the MiuraOriGeometry."""
        if self.visualize:
            studio = PiVizStudio(scene_fx=MiuraOriFX(self))
            studio.run()


if __name__ == '__main__':
    a = 1
    b = 1
    gamma = np.pi / 4
    theta = np.pi / 4
    xn = 5
    yn = 5

    geo = MiuraOriGeometry(a, b, gamma, theta, xn, yn, True)
    geo.spin()
