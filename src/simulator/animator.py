"""
animator.py

Modern visualization player for dynamic origami simulations.
Adapted from DEMLAT Visualization Player.

Author: Yogesh Phalak
"""

import numpy as np
from piviz import PiVizStudio, PiVizFX, pgfx, Colors, Colormap, Palette
from piviz.ui import Label, Button, Checkbox, Slider, Dropdown

class VisualizationScales:
    """Computes optimal visualization scales."""
    def __init__(self, geom, nodes):
        self.geom = geom
        # nodes shape: (i_max, j_max, 3)
        flat_pos = nodes.reshape(-1, 3)
        
        self.extent = np.max(flat_pos, axis=0) - np.min(flat_pos, axis=0)
        self.center = np.mean(flat_pos, axis=0)
        self.L_char = np.median(self.extent) if np.any(self.extent > 0) else 1.0
        
        # Heuristics
        self.node_radius = self.L_char * 0.02
        self.bar_radius = self.L_char * 0.01

class SimulationFX(PiVizFX):
    def __init__(self, geom, solution):
        self.geom = geom
        self.solution = solution
        self.x_sol = solution
        self.params = solution.sol
        self.dt = self.params[-1]
        
        self.i_max = geom.i_max
        self.j_max = geom.j_max
        self.n_nodes = self.i_max * self.j_max
        
        # Time series data
        # x_sol.y shape: (2 * n_nodes * 3, n_steps)
        # We only need positions for now: first half
        n_dof = self.n_nodes * 3
        self.positions_history = self.x_sol.y[:n_dof, :].T.reshape(-1, self.i_max, self.j_max, 3)
        self.times = self.x_sol.t
        self.n_frames = len(self.times)
        
        # Scales
        self.scales = VisualizationScales(geom, self.positions_history[0])
        
        # State
        self.timestep_idx = 0
        self.float_timestep = 0.0
        self.paused = True
        self.speed = 1.0
        
        # Config
        self.config = {
            'show_nodes': False,
            'show_bars': True,
            'show_faces': True,
            'show_strain': True,
            'strain_scale': 20.0,
            'colormap': 'jet',
            'bar_radius': self.scales.bar_radius,
            'node_radius': self.scales.node_radius,
            'face_opacity': 0.8,
            'use_spheres_for_nodes': True
        }
        
        # Pre-compute bar indices for faster access
        self.bar_data = []
        for bar in self.geom.bars:
            # Handle variable length bar definitions
            if len(bar) == 3:
                p0, p1, l0 = bar
            elif len(bar) == 4:
                p0, p1, l0, _ = bar # Ignore type
            else:
                continue # Skip unknown format
                
            self.bar_data.append({
                'p0_idx': tuple(p0),
                'p1_idx': tuple(p1),
                'l0': l0
            })
            
        # Pre-compute face indices
        self.faces = []
        
        # Check if geometry has a specific method to get faces (e.g. KirigamiSimpleCutGeometry)
        if hasattr(self.geom, 'get_faces'):
            self.faces = self.geom.get_faces()
        else:
            # Default grid faces
            for j in range(0, self.j_max - 1):
                for i in range(0, self.i_max - 1):
                    # Two triangles per grid cell
                    self.faces.append(((i, j), (i+1, j+1), (i+1, j)))
                    self.faces.append(((i, j), (i+1, j+1), (i, j+1)))

    def setup(self):
        if self.camera:
            self.camera.set_view('iso')
            self.camera.distance = self.scales.L_char * 3.0
            self.camera.target = tuple(self.scales.center)
            
        if self.ui_manager:
            self._setup_ui()
            
    def _setup_ui(self):
        self.ui_manager.set_panel_title("Origami Player")
        
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
            min_val=1.0, max_val=100.0,
            initial_val=self.config['strain_scale'],
            callback=self._set_strain_scale
        ))
        
        self.ui_manager.add_widget("drop_cmap", Dropdown(
            options=['jet', 'viridis', 'plasma', 'inferno', 'coolwarm'],
            selected_index=0,
            callback=lambda v: self.config.update({'colormap': v}),
            label="Colormap"
        ))
        
        self.ui_manager.add_widget("chk_faces", Checkbox(
            "Show Faces", self.config['show_faces'],
            lambda v: self.config.update({'show_faces': v})
        ))
        
        self.ui_manager.add_widget("sld_opacity", Slider(
            label="Face Opacity",
            min_val=0.0, max_val=1.0,
            initial_val=self.config['face_opacity'],
            callback=lambda v: self.config.update({'face_opacity': float(v)})
        ))
        
        self.ui_manager.add_widget("chk_bars", Checkbox(
            "Show Bars", self.config['show_bars'],
            lambda v: self.config.update({'show_bars': v})
        ))
        
        self.ui_manager.add_widget("chk_nodes", Checkbox(
            "Show Nodes", self.config['show_nodes'],
            lambda v: self.config.update({'show_nodes': v})
        ))
        
        self.ui_manager.add_widget("sld_node_radius", Slider(
            label="Node Radius",
            min_val=self.scales.node_radius * 0.1,
            max_val=self.scales.node_radius * 5.0,
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
        positions = self.positions_history[idx] # (i_max, j_max, 3)
        self.geom.nodes = positions # Update geom for consistency if needed
        
        # Render
        self._render_scene(positions)
        
    def _render_scene(self, positions):
        # Compute node strains for coloring
        node_strains = np.zeros((self.i_max, self.j_max))
        node_counts = np.zeros((self.i_max, self.j_max))
        
        if self.config['show_strain']:
            for bar in self.bar_data:
                p0 = positions[bar['p0_idx']]
                p1 = positions[bar['p1_idx']]
                
                # Check if l0 is valid (not None)
                if bar['l0'] is None:
                    continue

                l = np.linalg.norm(p0 - p1)
                strain = abs((l - bar['l0']) / bar['l0'])
                
                node_strains[bar['p0_idx']] += strain
                node_strains[bar['p1_idx']] += strain
                
                node_counts[bar['p0_idx']] += 1
                node_counts[bar['p1_idx']] += 1
            
            # Average the strains
            mask = node_counts > 0
            node_strains[mask] /= node_counts[mask]

        # Helper for color
        def get_color(val, alpha=1.0):
            s = val * self.config['strain_scale']
            t = np.clip(s, 0.0, 1.0)
            
            if self.config['colormap'] == 'jet':
                c = self._jet(t)
            elif self.config['colormap'] == 'viridis':
                c = Colormap.viridis(t)
            elif self.config['colormap'] == 'plasma':
                c = Colormap.plasma(t)
            elif self.config['colormap'] == 'inferno':
                c = Colormap.inferno(t)
            elif self.config['colormap'] == 'coolwarm':
                c = Colormap.coolwarm(t)
            else:
                c = (0.5, 0.5, 0.5)
                
            if len(c) == 3:
                return (c[0], c[1], c[2], alpha)
            return c

        # Render Faces
        if self.config['show_faces']:
            opacity = self.config['face_opacity']
            for face in self.faces:
                p0 = tuple(positions[face[0]])
                p1 = tuple(positions[face[1]])
                p2 = tuple(positions[face[2]])
                
                # Use draw_face for smooth interpolation
                # This requires per-vertex colors
                if self.config['show_strain']:
                    c0 = get_color(node_strains[face[0]], opacity)
                    c1 = get_color(node_strains[face[1]], opacity)
                    c2 = get_color(node_strains[face[2]], opacity)
                    pgfx.draw_face(p0, p1, p2, c0, c1, c2)
                else:
                    # Uniform color if strain is off
                    c = (0.9, 0.9, 0.9, opacity)
                    pgfx.draw_face(p0, p1, p2, c, c, c)

        # Render Bars
        if self.config['show_bars']:
            for bar in self.bar_data:
                p0 = tuple(positions[bar['p0_idx']])
                p1 = tuple(positions[bar['p1_idx']])
                
                color = (0.2, 0.2, 0.2)
                if self.config['show_strain']:
                    # Average strain of endpoints for bar color
                    s0 = node_strains[bar['p0_idx']]
                    s1 = node_strains[bar['p1_idx']]
                    color = get_color((s0 + s1) * 0.5)
                
                pgfx.draw_cylinder(
                    start=p0, end=p1, 
                    radius=self.config['bar_radius'], 
                    color=color, 
                    detail=6
                )
                
        # Render Nodes
        if self.config['show_nodes']:
            radius = self.config['node_radius']
            for j in range(self.j_max):
                for i in range(self.i_max):
                    pos = tuple(positions[i, j])
                    color = (0.3, 0.3, 0.3)
                    if self.config['show_strain']:
                        color = get_color(node_strains[i, j])
                    
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
    studio = PiVizStudio(scene_fx=SimulationFX(geom, solution))
    studio.run()
