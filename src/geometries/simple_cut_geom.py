"""
Kirigami Simple Cut Geometry

This code defines the KirigamiSimpleCutGeometry class, which represents a kirigami structure with simple cuts.
It provides methods to initialize the geometry, update nodes, add bars and hinges, and visualize the structure.

Author: Yogesh Phalak
Date: 2023-06-20
Filename: simple_cut_geom.py
"""

import numpy as np
import vpython as vp


class KirigamiSimpleCutGeometry:
    def __init__(self, a, b, cut_th, xn, yn, visualize=True):
        """
        Initialize the KirigamiSimpleCutGeometry.

        Parameters:
            a (float): Length parameter 'a' of the geometry.
            b (float): Length parameter 'b' of the geometry.
            cut_th (float): Cut angle in radians.
            xn (int): Number of divisions in the x-direction.
            yn (int): Number of divisions in the y-direction.
            visualize (bool, optional): Flag indicating whether to visualize the geometry. Defaults to True.
        """
        self.params = ['KirigamiSimpleCut', a, b, cut_th, xn, yn]
        self.a = a
        self.b = b
        self.cut_th = cut_th
        self.xn = xn
        self.yn = yn
        self.visualize = visualize
        self.fa = 1

        self.i_max = 2 * self.xn
        self.j_max = 2 * self.yn

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
        """
            Initialize the VPython scene for visualization.

            Sets up the VPython canvas with the appropriate settings.

            Parameters:
                None

            Returns:
                None
        """
        self.scene = vp.canvas(title='Kirigami Simple Cut Geometry', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(int(self.a * self.i_max / 2), int(self.b * self.j_max / 2), 0),
                               background=vp.color.white, fov=1)

    def update_nodes(self):
        """
            Update the node positions in the kirigami structure.

            Calculates and updates the positions of the nodes based on the current geometry parameters.

            Parameters:
                None

            Returns:
                None
        """
        self.fa = (self.a ** 2 + self.b ** 2) ** 0.5
        self.nodes = np.zeros((self.i_max, self.j_max, 3))

        for j in range(0, self.j_max):
            for i in range(0, int(self.i_max / 2)):
                self.nodes[i, j] = np.array([i * self.a, j * self.b, 0])

        for j in range(0, self.j_max):
            x = int(self.i_max / 2 - 1) * self.a + self.cut_th
            for i in range(int(self.i_max / 2), self.i_max):
                self.nodes[i, j] = np.array([x, j * self.b, 0])
                x += self.a

        if self.visualize:
            self.update_vp_nodes()

    def add_bar(self, p0, p1, l0):
        """
            Add a bar element to the kirigami structure.

            Defines a bar element with the given parameters and adds it to the structure.

            Parameters:
                p0 (list): Start node index (i, j) of the bar.
                p1 (list): End node index (i, j) of the bar.
                l0 (list): Rest length of the bar.

            Returns:
                None
        """
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        """
            Add a hinge element to the kirigami structure.

            Defines a hinge element with the given parameters and adds it to the structure.

            Parameters:
                p0 (list): First node index (i, j) of the hinge.
                p1 (list): Second node index (i, j) of the hinge.
                pa (list): Auxiliary node index (i, j) for the hinge.
                pb (list): Auxiliary node index (i, j) for the hinge.
                th0 (float): Rest angle of the hinge.
                l0 (float): Rest length of the hinge.
                type (str): Type of the hinge (e.g., "facet", "fold").

            Returns:
                None
        """
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

        # cuts
        [self.add_bar([int(self.i_max / 2) - 1, j], [int(self.i_max / 2) - 1, j + 1], np.linalg.norm(
            self.nodes[int(self.i_max / 2) - 1, j] - self.nodes[int(self.i_max / 2) - 1, j + 1])) for j in
         range(2, self.j_max - 3)]
        self.add_bar([int(self.i_max / 2) - 1, 2], [int(self.i_max / 2), 2],
                     np.linalg.norm(self.nodes[int(self.i_max / 2) - 1, 2] - self.nodes[int(self.i_max / 2), 2]))
        [self.add_bar([int(self.i_max / 2), j], [int(self.i_max / 2), j + 1],
                      np.linalg.norm(self.nodes[int(self.i_max / 2), j] - self.nodes[int(self.i_max / 2), j + 1])) for j
         in
         range(2, self.j_max - 3)]
        self.add_bar([int(self.i_max / 2) - 1, self.j_max - 3], [int(self.i_max / 2), self.j_max - 3], np.linalg.norm(
            self.nodes[int(self.i_max / 2) - 1, self.j_max - 3] - self.nodes[int(self.i_max / 2), self.j_max - 3]))

        # facets
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                if i == int(self.i_max / 2) - 1 and 1 < j < self.j_max - 3:
                    continue
                l0 = np.linalg.norm(self.nodes[i + 1, j + 1] - self.nodes[i, j])
                self.add_bar([i, j], [i + 1, j + 1], l0)
                self.add_hinge([i, j], [i + 1, j + 1],
                               [i, j + 1], [i + 1, j], np.pi, l0, 'facet')

        # folds
        for j in range(1, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                if i == int(self.i_max / 2) - 1 and 1 < j < self.j_max - 2:
                    continue
                l0 = np.linalg.norm(self.nodes[i + 1, j] - self.nodes[i, j])
                self.add_bar([i, j], [i + 1, j], l0)
                self.add_hinge([i, j], [i + 1, j], [i + 1, j + 1], [i, j - 1], np.pi, l0, 'fold')

        for j in range(0, self.j_max - 1):
            for i in range(1, self.i_max - 1):
                if int(self.i_max / 2) - 2 < i < int(self.i_max / 2) + 1 and 1 < j < self.j_max - 3:
                    continue
                l0 = np.linalg.norm(self.nodes[i, j + 1] - self.nodes[i, j])
                self.add_bar([i, j], [i, j + 1], l0)
                self.add_hinge([i, j], [i, j + 1], [i - 1, j], [i + 1, j + 1], np.pi, l0, 'fold')

    def draw(self):
        """
            Draw the kirigami structure.

            Draws the kirigami structure by creating triangles between the vertex nodes.

            Parameters:
                None

            Returns:
                None
        """
        if not self.visualize:
            return
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                if i == int(self.i_max / 2) - 1 and 1 < j < self.j_max - 3:
                    continue
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
        """
            Update the curves representing the kirigami structure.

            Updates the positions of the curves representing the kirigami structure based on the current node positions.

            Parameters:
                None

            Returns:
                None
        """
        n = 0
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                if i == int(self.i_max / 2) - 1 and 1 < j < self.j_max - 3:
                    continue
                self.crease_curves[n].modify(0, pos=self.vp_nodes[i][j].pos)
                self.crease_curves[n].modify(1, pos=self.vp_nodes[i + 1][j].pos)
                self.crease_curves[n].modify(2, pos=self.vp_nodes[i + 1][j + 1].pos)
                self.crease_curves[n].modify(3, pos=self.vp_nodes[i][j + 1].pos)
                self.crease_curves[n].modify(4, pos=self.vp_nodes[i][j].pos)
                self.facet_curves[n].modify(0, pos=self.vp_nodes[i][j].pos)
                self.facet_curves[n].modify(1, pos=self.vp_nodes[i + 1][j + 1].pos)
                n += 1

    def widget(self):
        """
            Create interactive widgets for adjusting the parameters.

            Creates sliders for adjusting the values of `a` and `b` parameters, and binds them to the respective
            slider handlers.

            Parameters:
                None

            Returns:
                None
        """
        vp.slider(min=0, max=2, value=self.a, bind=self.slider_theta)
        vp.slider(min=0, max=2, value=self.b, bind=self.slider_gamma)

    def slider_theta(self, a):
        self.a = a.value

    def slider_gamma(self, b):
        self.b = b.value

    def spin(self):
        """
            Start the spinning animation of the kirigami structure.

            Starts the spinning animation of the kirigami structure by continuously updating the node positions
            and refreshing the visualization.

            Parameters:
                None

            Returns:
                None
        """
        if self.visualize:
            self.widget()
            while True:
                self.update_nodes()
                vp.rate(30)


if __name__ == '__main__':
    a = 0.3
    b = 0.3
    cut_th = 0.05
    xn = 12
    yn = 12

    geo = KirigamiSimpleCutGeometry(a, b, cut_th, xn, yn, True)
    geo.spin()
