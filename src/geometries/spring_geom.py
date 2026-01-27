import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors
from piviz.ui import Label, Slider


def spring_curve(d, p, th, t):
    z = t / (2 * np.pi) * p
    x = (d / 2 - z * np.tan(th)) * np.cos(t)
    y = (d / 2 - z * np.tan(th)) * np.sin(t)
    return x, y, z


class SpringFX(PiVizFX):
    def __init__(self, geometry):
        self.geometry = geometry

    def setup(self):
        if self.ui_manager:
            self.ui_manager.set_panel_title("Spring Geometry")
            self.ui_manager.add_widget("info", Label("Adjust parameters:", color=(1, 1, 1, 1.0)))
            
            self.ui_manager.add_widget("slider_p", Slider(
                label="Pitch (p)",
                min_val=0.0, max_val=self.geometry.d,
                initial_val=self.geometry.p,
                callback=self._update_p
            ))
            
            self.ui_manager.add_widget("slider_th", Slider(
                label="Taper Angle (th)",
                min_val=0.0, max_val=np.pi / 4,
                initial_val=self.geometry.th,
                callback=self._update_th
            ))
        
        if self.camera:
            self.camera.distance = 5.0
            self.camera.set_view('iso')

    def _update_p(self, value):
        self.geometry.p = float(value)

    def _update_th(self, value):
        self.geometry.th = float(value)

    def render(self, time, dt):
        self.geometry.update_geometry()
        self.geometry.draw_piviz()


class SpringGeometry:

    def __init__(self, d, p, n, th, visualize=True, res=100):
        self.params = ['SpringGeometry', d, p, n, th]
        self.d = d
        self.p = p
        self.th = th
        self.n = int(n)
        self.visualize = visualize
        self.res = res
        self.l = 0

        self.x = self.get_spring_shape()

        self.i_max = self.x.shape[0]
        self.x0 = self.x.copy()

    def get_spring_shape(self):
        t = np.linspace(0, 2 * self.n * np.pi, self.res)
        l = np.trapz(
            ((self.d / 2 - self.p * np.tan(self.th) / (2 * np.pi) * t) ** 2 + (self.p / (2 * np.pi)) ** 2) ** 0.5, t)
        ls = np.linspace(0, l, self.res)
        ts = np.zeros_like(ls)
        for i in range(1, self.res):
            ts[i] = ts[i - 1] + l / (
                    self.res * ((self.d / 2 - self.p * np.tan(self.th) / (2 * np.pi) * ts[i - 1]) ** 2 + (
                    self.p / (2 * np.pi)) ** 2) ** 0.5)
        x = np.zeros((self.res, 3))
        x[:, 0], x[:, 1], x[:, 2] = spring_curve(self.d, self.p, self.th, ts)
        self.l = l
        return x

    def update_geometry(self):
        self.x = self.get_spring_shape()
        self.i_max = self.x.shape[0]
        self.x0 = self.x.copy()

    def draw_piviz(self):
        if not self.visualize:
            return
            
        # Draw spring as a path
        points = [tuple(p) for p in self.x]
        pgfx.draw_path(points, color=Colors.WHITE, width=2.0)
        
        # Draw nodes as small spheres
        for p in self.x:
            pgfx.draw_sphere(center=tuple(p), radius=self.d / 40, color=Colors.WHITE, detail=6)

    def spin(self):
        if self.visualize:
            studio = PiVizStudio(scene_fx=SpringFX(self))
            studio.run()


if __name__ == '__main__':
    spring = SpringGeometry(0.3, 0.03, 10, 10 * np.pi / 180, True)
    print(spring.l)
    spring.spin()
