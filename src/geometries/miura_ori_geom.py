"""
MiuraOriGeometry Class

This class represents the geometry of a Miura-Ori origami structure. It provides methods to initialize the geometry,
update dimensions, update nodes, add bars and hinges, and visualize the geometry using VPython.

Author: Yogesh Phalak
Date: 2023-06-20
Filename: miura_ori_geom.py

"""

import numpy as np
import vpython as vp


class MiuraOriGeometry:
    def __init__(self, a, b, gamma, theta, xn, yn, visualize=True):
        """Initialize the MiuraOriGeometry instance with specified parameters.

            The __init__ method is called when creating a new instance of the MiuraOriGeometry class.
            It initializes the attributes of the instance based on the provided parameters.
            It also sets up the necessary variables and arrays for the geometry calculations.

            Args:
                a (float): Length of the base unit square.
                b (float): Width of the base unit square.
                gamma (float): Angle between the crease lines.
                theta (float): Twist angle of the crease pattern.
                xn (int): Number of unit squares along the x-axis.
                yn (int): Number of unit squares along the y-axis.
                visualize (bool, optional): Flag to enable visualization. Defaults to True.

            Returns:
                None
        """
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
        """Initialize the visualization scene.

            The init_scene method creates a new canvas with specified dimensions and settings for the Miura-Ori geometry
             visualization.

            Returns:
                None
        """
        self.scene = vp.canvas(title='Miura-Ori Geometry', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(self.l * self.xn, self.w * self.yn, 0),
                               background=vp.color.white, fov=1)

    def update_dimensions(self):
        """Update the dimensions of the Miura-Ori geometry.

           The update_dimensions method calculates and updates the various dimensions of the Miura-Ori geometry based
           on the provided parameters.

           Returns:
               None
        """
        self.ht = self.a * np.sin(self.gamma) * np.sin(self.theta)
        self.t = self.a * np.sin(self.gamma) * np.sin(self.theta)
        self.l = self.b * np.tan(self.gamma) * np.cos(self.theta) / (
                (1 + np.cos(self.theta) ** 2 * np.tan(self.gamma) ** 2) ** 0.5)
        self.w = self.a * (1 - np.sin(self.theta) ** 2 * np.sin(self.gamma) ** 2) ** 0.5
        self.v = self.b / (1 + np.cos(self.theta) ** 2 * np.tan(self.gamma) ** 2) ** 0.5
        self.fa = (self.a ** 2 + self.b ** 2 - 2 * self.a * self.b * np.cos(np.pi - self.gamma)) ** 0.5

    def update_nodes(self):
        """Update the positions of the nodes in the Miura-Ori geometry.

          The update_nodes method recalculates and updates the positions of the nodes in the Miura-Ori geometry based
          on the current dimensions and parameters.

          Returns:
              None
        """
        self.nodes = np.zeros((self.i_max, self.j_max, 3))
        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.nodes[i, j] = np.array([i * self.l, j * self.w + self.v * (i % 2), self.ht * (j % 2)])
        if self.visualize:
            self.update_vp_nodes()

    def add_bar(self, p0, p1, l0):
        """
            Add a bar to the geometry.

            Parameters:
                p0 (list[int]): Index of the first node
                p1 (list[int]): Index of the second node
                l0 (float): Rest length of the bar
        """
        self.bars.append([p0, p1, l0])

    def add_hinge(self, p0, p1, pa, pb, th0, l0, type):
        """
            Add a hinge to the geometry.

            Parameters:
                p0 (list[int]): Index of the first node
                p1 (list[int]): Index of the second node
                pa (list[int]): Index of the third node
                pb (list[int]): Index of the fourth node
                th0 (float): Rest angle of the hinge
                l0 (float): Rest length of the hinge
                type (str): Type of the hinge ('facet' or 'fold')
        """
        self.hinges.append([p0, p1, pa, pb, th0, l0, type])

    def update_vp_nodes(self, colors=None):
        """
            Update the VPython nodes based on the current node positions.

            Parameters:
                colors (list[list[vp.color]]): Colors for the nodes (optional)
        """
        for j in range(0, self.j_max):
            for i in range(0, self.i_max):
                self.vp_nodes[i][j].pos = vp.vector(self.nodes[i, j, 1], self.nodes[i, j, 0], self.nodes[i, j, 2])
                if colors is not None:
                    self.vp_nodes[i][j].color = colors[i][j]
        self.update_curves()

    def add_bar_hinges(self):
        """
            Add bars and hinges to the geometry based on the current parameters.
        """
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

    def draw(self):
        """
            Draw the geometry using VPython.
        """
        if not self.visualize:
            return
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                [vp.triangle(vs=[self.vp_nodes[i][j + i % 2],
                                 self.vp_nodes[i + 1][j + 1 - i % 2],
                                 self.vp_nodes[i + p][j + q]],
                             color=vp.color.red) for p in range(0, 2) for q in range(0, 2)]

        self.crease_curves = [[vp.curve(pos=[self.vp_nodes[i][j].pos,
                                             self.vp_nodes[i + 1][j].pos,
                                             self.vp_nodes[i + 1][j + 1].pos,
                                             self.vp_nodes[i][j + 1].pos,
                                             self.vp_nodes[i][j].pos],
                                        color=vp.color.gray(0.0),
                                        radius=0.01)
                               for j in range(0, self.j_max - 1)] for i in range(0, self.i_max - 1)]

        self.facet_curves = [[vp.curve(pos=[self.vp_nodes[i][j + i % 2].pos,
                                            self.vp_nodes[i + 1][j + 1 - i % 2].pos],
                                       color=vp.color.black, radius=0.01)
                              for j in range(0, self.j_max - 1)] for i in range(0, self.i_max - 1)]

    def update_curves(self):
        """
           Update the VPython curves based on the current node positions.
        """
        for i in range(0, self.i_max - 1):
            for j in range(0, self.j_max - 1):
                self.crease_curves[i][j].modify(0, pos=self.vp_nodes[i][j].pos)
                self.crease_curves[i][j].modify(1, pos=self.vp_nodes[i + 1][j].pos)
                self.crease_curves[i][j].modify(2, pos=self.vp_nodes[i + 1][j + 1].pos)
                self.crease_curves[i][j].modify(3, pos=self.vp_nodes[i][j + 1].pos)
                self.crease_curves[i][j].modify(4, pos=self.vp_nodes[i][j].pos)
                self.facet_curves[i][j].modify(0, pos=self.vp_nodes[i][j + i % 2].pos)
                self.facet_curves[i][j].modify(1, pos=self.vp_nodes[i + 1][j + 1 - i % 2].pos)

    def widget(self):
        """Creates and displays sliders for adjusting theta and gamma parameters.

            The widget function creates two sliders using the `vp.slider` function from the `vpython` module.
            The sliders allow the user to interactively adjust the values of the theta and gamma parameters,
            which control the geometry of the MiuraOri structure. The minimum value of both sliders is set to 0,
            and the maximum value is set to pi/2. The initial values of the sliders are set to the current values of the
            theta and gamma attributes of the MiuraOriGeometry instance.

            Args:
                self: The MiuraOriGeometry instance.

            Returns:
                None
        """
        vp.slider(min=0, max=np.pi / 2, value=self.theta, bind=self.slider_theta)
        vp.slider(min=0, max=np.pi / 2, value=self.gamma, bind=self.slider_gamma)

    def slider_theta(self, theta):
        """Callback function for the theta slider.

            The slider_theta function is a callback function that gets called when the value of the theta slider is changed.
            It updates the value of the theta attribute in the MiuraOriGeometry instance with the new value from the slider.

            Args:
                self: The MiuraOriGeometry instance.
                theta: The new value of the theta slider.

            Returns:
                None
        """
        self.theta = theta.value

    def slider_gamma(self, gamma):
        """Callback function for the gamma slider.

           The slider_gamma function is a callback function that gets called when the value of the gamma slider is changed.
            It updates the value of the gamma attribute in the MiuraOriGeometry instance with the new value from the slider.

           Args:
               self: The MiuraOriGeometry instance.
               gamma: The new value of the gamma slider.

           Returns:
               None
        """
        self.gamma = gamma.value

    def spin(self):
        """Starts the interactive animation of the MiuraOriGeometry.

            The spin function is responsible for starting the interactive animation of the MiuraOriGeometry.
            If the visualize attribute is set to True, it creates a widget and enters a loop where it continuously
            updates the dimensions and nodes of the geometry based on the current values of theta and gamma.
            It also uses the vp.rate function from the `vpython` module to control the animation speed.

            Args:
                self: The MiuraOriGeometry instance.

            Returns:
                None
        """
        if self.visualize:
            self.widget()
            while True:
                self.update_dimensions()
                self.update_nodes()
                vp.rate(30)


if __name__ == '__main__':
    a = 1
    b = 1
    gamma = np.pi / 4
    theta = np.pi / 4
    xn = 2
    yn = 2

    geo = MiuraOriGeometry(a, b, gamma, theta, xn, yn, True)
    geo.spin()
