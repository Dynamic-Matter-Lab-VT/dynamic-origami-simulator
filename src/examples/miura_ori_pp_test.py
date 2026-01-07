import numpy as np
from phalcopulse import PhalcoPulseStudio, PhalcoPulseFX, pgfx
from phalcopulse.ui import Label, Slider


class MiuraOriFX(PhalcoPulseFX):
    def setup(self, ui_manager):
        # --- Parameters ---
        self.a = 1.0
        self.b = 1.0
        self.xn = 4  # Increased grid size since PhalcoPulse can handle it
        self.yn = 4

        # Initial angles
        self.gamma = np.pi / 4
        self.theta = np.pi / 4

        # State variables
        self.nodes = None
        self.l, self.w, self.v, self.ht = 0, 0, 0, 0

        # --- UI Callbacks ---
        def set_theta(val):
            self.theta = val
            self.update_label()

        def set_gamma(val):
            self.gamma = val
            self.update_label()

        # --- UI Layout ---
        # Title
        ui_manager.add_widget("title", Label((15, 200, 300, 20), "-- Miura-Ori Generator --", align='left'))

        # Theta Control
        ui_manager.add_widget("lbl_theta", Label((15, 170, 200, 20), "Theta (Twist)", align='left'))
        ui_manager.add_widget("sl_theta", Slider((15, 150, 200, 20), "", 0.1, 1.5, self.theta, set_theta))

        # Gamma Control
        ui_manager.add_widget("lbl_gamma", Label((15, 120, 200, 20), "Gamma (Crease)", align='left'))
        ui_manager.add_widget("sl_gamma", Slider((15, 100, 200, 20), "", 0.1, 1.5, self.gamma, set_gamma))

        # Status Label
        self.status_label = ui_manager.add_widget("status", Label((15, 50, 300, 20), "", align='left'))
        self.update_label()

    def update_label(self):
        self.status_label.text = f"T: {self.theta:.2f} | G: {self.gamma:.2f}"

    def update_geometry(self):
        """
        Recalculates the geometry.
        Uses NumPy vectorization for performance.
        """
        # Calculate Dimensions
        sin_g, cos_g = np.sin(self.gamma), np.cos(self.gamma)
        sin_t, cos_t = np.sin(self.theta), np.cos(self.theta)
        tan_g = np.tan(self.gamma)

        self.ht = self.a * sin_g * sin_t
        self.l = self.b * tan_g * cos_t / ((1 + cos_t ** 2 * tan_g ** 2) ** 0.5)
        self.w = self.a * (1 - sin_t ** 2 * sin_g ** 2) ** 0.5
        self.v = self.b / (1 + cos_t ** 2 * tan_g ** 2) ** 0.5

        # Grid dimensions
        i_max = 2 * self.xn + 1
        j_max = 2 * self.yn + 1

        # Vectorized Grid Generation
        i = np.arange(i_max)
        j = np.arange(j_max)
        I, J = np.meshgrid(i, j, indexing='ij')

        # Calculate Coordinates
        X = I * self.l
        Y = J * self.w + self.v * (I % 2)
        Z = self.ht * (J % 2)

        # Center the model
        center_x = (i_max * self.l) / 2
        center_y = (j_max * self.w) / 2

        self.nodes = np.stack((X - center_x, Z, Y - center_y), axis=-1)  # Swapped Y/Z for typical 3D Up-axis

    def loop(self, delta_time):
        # Update math every frame (or you could flag it to only update on change)
        self.update_geometry()

        i_max, j_max, _ = self.nodes.shape

        # --- Draw Facets (Triangles) ---
        # Note: In immediate mode, we loop and draw.
        # For huge grids, we'd want a batch mesh, but this works for N < 50.
        for i in range(i_max - 1):
            for j in range(j_max - 1):
                # Determine quad vertices
                p00 = self.nodes[i, j]
                p10 = self.nodes[i + 1, j]
                p01 = self.nodes[i, j + 1]
                p11 = self.nodes[i + 1, j + 1]

                # Pattern logic for split direction
                if (i % 2) == 0:
                    # Draw two triangles for the quad
                    pgfx.draw_triangle(p00, p10, p11, color=(0.8, 0.2, 0.2))  # Red-ish
                    pgfx.draw_triangle(p00, p11, p01, color=(0.8, 0.2, 0.2))
                else:
                    # Alternate triangulation if needed, or keep consistent
                    pgfx.draw_triangle(p00, p10, p01, color=(0.8, 0.2, 0.2))
                    pgfx.draw_triangle(p10, p11, p01, color=(0.8, 0.2, 0.2))

        # --- Draw Creases (Using thin cylinders) ---
        # Vertical-ish lines
        for i in range(i_max):
            for j in range(j_max - 1):
                pgfx.draw_cylinder(
                    start=self.nodes[i, j],
                    end=self.nodes[i, j + 1],
                    radius=0.02,
                    color=(0.1, 0.1, 0.1)
                )

        # Horizontal-ish lines
        for j in range(j_max):
            for i in range(i_max - 1):
                pgfx.draw_cylinder(
                    start=self.nodes[i, j],
                    end=self.nodes[i + 1, j],
                    radius=0.02,
                    color=(0.1, 0.1, 0.1)
                )


if __name__ == '__main__':
    studio = PhalcoPulseStudio(scene_fx=MiuraOriFX())
    studio.run()
