import numpy as np
import vpython as vp


class SpringGeometry:

    def __init__(self, d, p, n, th, visualize=True):
        self.params = ['SpringGeometry', d, p, n, th]
        self.d = d
        self.p = p
        self.th = th
        self.n = int(n)
        self.visualize = visualize

        self.x = self.get_spring_shape()

        self.i_max = self.x.shape[0]
        self.x0 = self.x.copy()

        if self.visualize:
            self.scene = None
            self.init_scene()
            self.vp_nodes = self.create_vp_nodes()
            self.spring_curves = []
            self.draw()

    def get_spring_shape(self):
        m = int(50 * self.d)
        x = np.zeros((m * self.n, 3))
        for i in range(self.n):
            for j in range(m):
                idx = i * m + j
                x[idx, 0] = (self.d / 2 - (i + j / m) * self.p * np.tan(self.th)) * np.cos(2 * np.pi * j / m)
                x[idx, 1] = (self.d / 2 - (i + j / m) * self.p * np.tan(self.th)) * np.sin(2 * np.pi * j / m)
                x[idx, 2] = (i + j / m) * self.p

        return x

    def create_vp_nodes(self):
        vp_nodes = []
        for xn in self.x:
            vp_nodes.append(vp.vertex(pos=vp.vector(xn[0], xn[1], xn[2]), color=vp.color.white))
        return vp_nodes

    def init_scene(self):
        self.scene = vp.canvas(title='Spring Geometry', width=1000, height=800,
                               x=0,
                               y=0,
                               center=vp.vector(0, 0, 0),
                               background=vp.color.black, fov=1)

    def draw(self):
        for i in range(self.i_max - 1):
            self.spring_curves.append(
                vp.curve(pos=[vp.vector(self.x[i, 0], self.x[i, 1], self.x[i, 2]),
                              vp.vector(self.x[i + 1, 0], self.x[i + 1, 1], self.x[i + 1, 2])],
                         color=vp.color.white, radius=self.d / 40))

    def update_vp_nodes(self, colors=None):
        for i in range(self.i_max):
            self.vp_nodes[i].pos = vp.vector(self.x[i, 0], self.x[i, 1], self.x[i, 2])
            if colors is not None:
                self.vp_nodes[i].color = colors[i]

    def update_spring_curves(self):
        for i in range(self.i_max - 1):
            self.spring_curves[i].modify(0, self.vp_nodes[i].pos)
            self.spring_curves[i].modify(1, self.vp_nodes[i + 1].pos)
            self.spring_curves[i].color = self.vp_nodes[i].color

    def update_geometry(self):
        self.x = self.get_spring_shape()
        self.i_max = self.x.shape[0]
        self.x0 = self.x.copy()
        self.vp_nodes = self.create_vp_nodes()
        self.update_vp_nodes()
        self.update_spring_curves()

    def widget(self):
        self.scene.append_to_caption("""\n Instructions:
        Drag mouse or one finger to rotate.
        Pinch or two fingers to zoom.\n\n\n""")
        self.scene.append_to_caption("""pitch (p): """)
        vp.slider(min=0, max=5, value=0.1, bind=self.slider_p)
        self.scene.append_to_caption("""\n""")
        self.scene.append_to_caption("""taper angle (th): """)
        vp.slider(min=0, max=np.pi / 4, value=0, bind=self.slider_th)
        self.scene.append_to_caption("""\n""")

    def slider_p(self, p):
        self.p = p.value

    def slider_th(self, th):
        self.th = th.value

    def slider_n(self, n):
        self.n = int(n.value)

    def slider_d(self, d):
        self.d = d.value

    def spin(self):
        if self.visualize:
            self.widget()
            while True:
                self.update_geometry()
                vp.rate(10)


if __name__ == '__main__':
    spring = SpringGeometry(0.3, 0.06, 10, 10 * np.pi / 180, True)
    spring.spin()
