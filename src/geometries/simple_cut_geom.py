"""
Kirigami Simple Cut Geometry

This code defines the KirigamiSimpleCutGeometry class, which represents a kirigami structure with simple cuts.
It provides methods to initialize the geometry, update nodes, add bars and hinges, and visualize the structure.

Author: Yogesh Phalak
Date: 2023-06-20
Filename: simple_cut_geom.py
"""

import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


class KirigamiSimpleCutFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Kirigami Simple Cut Geometry")
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
            self.camera.distance = 15.0
            self.camera.set_view('iso')

    def _update_a(self, value):
        self.geometry.a = float(value)

    def _update_b(self, value):
        self.geometry.b = float(value)

    def render(self, time, dt):
        self.geometry.update_nodes()
        self.geometry.draw_piviz()


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

        self.update_nodes()
        self.add_bar_hinges()

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
        # Deprecated
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

    def get_faces(self):
        """Returns a list of faces (triangles) for visualization."""
        faces = []
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                # Skip the cut area
                if i == int(self.i_max / 2) - 1 and 1 < j < self.j_max - 3:
                    continue
                
                # Triangle 1: (i, j), (i+1, j+1), (i+1, j)
                faces.append(((i, j), (i+1, j+1), (i+1, j)))
                # Triangle 2: (i, j), (i+1, j+1), (i, j+1)
                faces.append(((i, j), (i+1, j+1), (i, j+1)))
        return faces

    def draw_piviz(self):
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
        
        faces = self.get_faces()
        for face in faces:
            v0 = tuple(self.nodes[face[0]])
            v1 = tuple(self.nodes[face[1]])
            v2 = tuple(self.nodes[face[2]])
            pgfx.draw_triangle(v0, v1, v2, color=Colors.RED)

        # Crease curves (Gray)
        # Re-using the logic from get_faces to draw edges might be redundant if we just want visual feedback
        # But let's keep the original outline logic
        for j in range(0, self.j_max - 1):
            for i in range(0, self.i_max - 1):
                if i == int(self.i_max / 2) - 1 and 1 < j < self.j_max - 3:
                    continue
                
                v00 = tuple(self.nodes[i, j])
                v10 = tuple(self.nodes[i + 1, j])
                v01 = tuple(self.nodes[i, j + 1])
                v11 = tuple(self.nodes[i + 1, j + 1])
                
                pgfx.draw_path([v00, v10, v11, v01, v00], color=(0.5, 0.5, 0.5), width=1.0)
                pgfx.draw_line(v00, v11, color=Colors.BLACK, width=1.0)

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
            studio = PiVizStudio(scene_fx=KirigamiSimpleCutFX(self))
            studio.run()


if __name__ == '__main__':
    a = 0.3
    b = 0.3
    cut_th = 0.05
    xn = 12
    yn = 12

    geo = KirigamiSimpleCutGeometry(a, b, cut_th, xn, yn, True)
    geo.spin()
