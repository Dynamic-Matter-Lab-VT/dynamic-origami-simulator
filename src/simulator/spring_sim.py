"""
Spring Simulation Player

This code loads and plays simulation results generated from the spring model solver. It visualizes the simulation
using PiViz.

Author: Yogesh Phalak
Date: 2023-06-20
Filename: spring_sim.py
"""

import time
import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Colormap
from piviz.ui import Label, Button, Checkbox, Slider, Dropdown


class SpringSimulationFX(PiVizFX):
    def __init__(self, geom, solution):
        self.geom = geom
        self.solution = solution
        self.x_sol = solution
        self.params = solution.sol
        self.dt = self.params[-1]
        
        self.i_max = geom.i_max
        
        # Time series data
        # x_sol.y shape: (2 * n_nodes * 3, n_steps)
        # We only need positions for now: first half
        n_dof = self.i_max * 3
        self.positions_history = self.x_sol.y[:n_dof, :].T.reshape(-1, self.i_max, 3)
        self.times = self.x_sol.t
        self.n_frames = len(self.times)
        
        # Initial positions for strain calculation
        self.x0 = geom.x0
        
        # State
        self.timestep_idx = 0
        self.float_timestep = 0.0
        self.paused = True
        self.speed = 1.0
        
        # Config
        self.config = {
            'show_strain': False,
            'show_nodes': False,
            'strain_scale': 20.0,
            'colormap': 'jet',
            'bar_radius': geom.d / 40,
            'node_radius': geom.d / 20
        }

    def setup(self):
        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = self.geom.d * 5.0
            self.camera.target = (0, 0, self.geom.l / 2)
            
        if self.ui_manager:
            self._setup_ui()
            
    def _setup_ui(self):
        self.ui_manager.set_panel_title("Spring Simulation Player")
        
        # Info
        self.lbl_time = Label(f"Time: 0.00s | Frame: 0/{self.n_frames}")
        self.ui_manager.add_widget("lbl_time", self.lbl_time)
        
        # Timeline
        self.sld_timeline = Slider(
            label="Timeline",
            min_val=0.0,
            max_val=float(self.n_frames - 1),
            initial_val=0.0,
            callback=self._on_timeline_change
        )
        self.ui_manager.add_widget("sld_timeline", self.sld_timeline)
        
        # Controls
        self.btn_pause = Button("Play", self._toggle_pause)
        
        self.ui_manager.add_widget("btn_prev", Button("<<", self._prev_frame))
        self.ui_manager.add_widget("btn_pause", self.btn_pause)
        self.ui_manager.add_widget("btn_next", Button(">>", self._next_frame))
        
        self.ui_manager.add_widget("sld_speed", Slider(
            label="Speed",
            min_val=0.1, max_val=5.0,
            initial_val=self.speed,
            callback=self._set_speed
        ))
        
        # Visualization
        self.ui_manager.add_widget("lbl_viz", Label("Visualization", color=(0.7, 0.9, 1.0)))
        
        self.ui_manager.add_widget("chk_strain", Checkbox(
            "Show Strain", self.config['show_strain'],
            lambda v: self.config.update({'show_strain': v})
        ))
        
        self.ui_manager.add_widget("sld_strain_scale", Slider(
            label="Strain Scale",
            min_val=1.0, max_val=50000.0,
            initial_val=self.config['strain_scale'],
            callback=self._set_strain_scale
        ))
        
        self.ui_manager.add_widget("drop_cmap", Dropdown(
            options=['jet', 'viridis', 'plasma', 'inferno', 'coolwarm'],
            selected_index=0,
            callback=lambda v: self.config.update({'colormap': v}),
            label="Colormap"
        ))
        
        self.ui_manager.add_widget("chk_nodes", Checkbox(
            "Show Nodes", self.config['show_nodes'],
            lambda v: self.config.update({'show_nodes': v})
        ))
        
        self.ui_manager.add_widget("sld_node_radius", Slider(
            label="Node Radius",
            min_val=self.config['node_radius'] * 0.1,
            max_val=self.config['node_radius'] * 5.0,
            initial_val=self.config['node_radius'],
            callback=lambda v: self.config.update({'node_radius': float(v)})
        ))

    # Callbacks
    def _on_timeline_change(self, value):
        self.timestep_idx = int(value)
        self.float_timestep = float(value)
        self.paused = True
        self._update_pause_text()
        
    def _prev_frame(self):
        self.paused = True
        self._update_pause_text()
        self.timestep_idx = max(0, self.timestep_idx - 5)
        self.float_timestep = float(self.timestep_idx)
        self._sync_slider()
        
    def _next_frame(self):
        self.paused = True
        self._update_pause_text()
        self.timestep_idx = min(self.n_frames - 1, self.timestep_idx + 5)
        self.float_timestep = float(self.timestep_idx)
        self._sync_slider()
        
    def _toggle_pause(self):
        self.paused = not self.paused
        self._update_pause_text()
        
    def _update_pause_text(self):
        if self.btn_pause:
            self.btn_pause.text = "Play" if self.paused else "Pause"
        
    def _set_speed(self, value):
        self.speed = float(value)
        
    def _set_strain_scale(self, value):
        self.config['strain_scale'] = float(value)
        
    def _sync_slider(self):
        if self.sld_timeline:
            self.sld_timeline.set_value(float(self.timestep_idx))

    def render(self, time_val, dt):
        # Update playback
        if not self.paused:
            self.float_timestep += dt * 30 * self.speed # 30 FPS base
            if self.float_timestep >= self.n_frames:
                self.float_timestep = 0.0
            self.timestep_idx = int(self.float_timestep)
            self._sync_slider()
            
        idx = min(self.timestep_idx, self.n_frames - 1)
        
        # Update UI label
        t_curr = self.times[idx]
        if self.lbl_time:
            status = "PAUSED" if self.paused else "PLAYING"
            self.lbl_time.text = f"Time: {t_curr:.3f}s | Frame: {idx}/{self.n_frames} | {status}"
            
        # Get positions
        positions = self.positions_history[idx] # (i_max, 3)
        self.geom.x = positions # Update geom for consistency if needed
        
        # Render
        self._render_scene(positions)
        
    def _render_scene(self, positions):
        # Compute strains for coloring
        strains = np.zeros(self.i_max)
        
        if self.config['show_strain']:
            for i in range(self.i_max - 1):
                l_curr = np.linalg.norm(positions[i] - positions[i+1])
                l_orig = np.linalg.norm(self.x0[i] - self.x0[i+1])
                strain = abs(l_curr / l_orig - 1)
                strains[i] = strain
            strains[-1] = strains[-2] # Last node gets same strain as last segment

        # Helper for color
        def get_color(val):
            s = val * self.config['strain_scale']
            t = np.clip(s, 0.0, 1.0)
            
            if self.config['colormap'] == 'jet':
                return self._jet(t)
            elif self.config['colormap'] == 'viridis':
                return Colormap.viridis(t)
            elif self.config['colormap'] == 'plasma':
                return Colormap.plasma(t)
            elif self.config['colormap'] == 'inferno':
                return Colormap.inferno(t)
            elif self.config['colormap'] == 'coolwarm':
                return Colormap.coolwarm(t)
            return (1.0, 1.0, 1.0)

        # Render Spring Segments
        for i in range(self.i_max - 1):
            p0 = tuple(positions[i])
            p1 = tuple(positions[i+1])
            
            color = (1.0, 1.0, 1.0)
            if self.config['show_strain']:
                color = get_color(strains[i])
            
            pgfx.draw_cylinder(
                start=p0, end=p1, 
                radius=self.config['bar_radius'], 
                color=color, 
                detail=6
            )
            
        # Render Nodes
        if self.config['show_nodes']:
            radius = self.config['node_radius']
            for i in range(self.i_max):
                pos = tuple(positions[i])
                color = (0.3, 0.3, 0.3)
                if self.config['show_strain']:
                    # Use average strain of adjacent segments for node color
                    s_left = strains[i-1] if i > 0 else strains[0]
                    s_right = strains[i] if i < self.i_max - 1 else strains[-1]
                    color = get_color((s_left + s_right) * 0.5)
                
                pgfx.draw_sphere(
                    center=pos,
                    radius=radius,
                    color=color,
                    detail=8
                )

    def _jet(self, v):
        v = np.clip(v, 0, 1)
        if v < 0.25:
            return (0, v*4, 1)
        elif v < 0.5:
            return (0, 1, 1-(v-0.25)*4)
        elif v < 0.75:
            return ((v-0.5)*4, 1, 0)
        else:
            return (1, 1-(v-0.75)*4, 0)

def simulate(geom, solution):
    studio = PiVizStudio(scene_fx=SpringSimulationFX(geom, solution))
    studio.run()
