"""
animator

This code loads and plays simulation results generated from the dynamic model solver. It visualizes the simulation
using VPython.

Author: Yogesh Phalak
Date: 2023-06-20
Filename: animator.py

"""
import time
import numpy as np
import vpython as vp
import matplotlib.pyplot as plt

show_strain = False
paused = False
strain_scale = 20
time_scale = 1
cmap = 'jet'


def interpolate_color(value, cmap='jet'):
    """
        Interpolates a color between blue and green for values less than or equal to 0.5,
        and between green and red for values greater than 0.5.

        Args:
            value (float): The value between 0 and 1 used to interpolate the color.

        Returns:
            vpython.vector: The interpolated color as a VPython vector.

    """
    cm = plt.get_cmap(cmap)
    return vp.vector(*cm(value)[:3])


def pause_button_func(p):
    """
        Function to handle the pause/play button action.

        Args:
            p: The button object.

        Returns:
            None
    """
    global paused, pause_button
    paused = not paused
    if paused:
        pause_button.text = 'Play'
    else:
        pause_button.text = 'Pause'


def strain_checkbox_func(s):
    """
        Function to handle the strain checkbox action.

        Args:
            s: The checkbox object.

        Returns:
            None
    """
    global show_strain, strain_checkbox
    show_strain = s.checked


def slider_strain_scale(s):
    """
        Function to handle the strain scale slider action.

        Args:
            s: The slider object.

        Returns:
            None
    """
    global strain_scale
    strain_scale = s.value


def slider_time_scale(s):
    """
        Function to handle the timescale slider action.

        Args:
            s: The slider object.

        Returns:
            None
    """
    global time_scale
    time_scale = s.value


def update_sim_time(t):
    pass


def cmap_update(m):
    global cmap
    cmap = m.selected
    print(cmap)


def simulate(geom_, solution_):
    global show_strain, paused, strain_scale, time_scale, pause_button, strain_checkbox, slider1, slider2, text, curr_time, time_slider, geom, cmap
    x_sol = solution_
    params = x_sol.sol
    geom = geom_
    dt = params[-1]

    i_max = geom.i_max
    j_max = geom.j_max

    geom.scene.append_to_caption("""\n""")
    time_slider = vp.slider(min=0, max=x_sol.t[-1], value=0, bind=update_sim_time, textwrap=True, length=1000)
    geom.scene.append_to_caption("""\n""")

    pause_button = vp.button(text='Pause', bind=pause_button_func)
    geom.scene.append_to_caption("""\n\n""")
    strain_checkbox = vp.checkbox(text='Show Strain', bind=strain_checkbox_func)
    # add dropdown list for strain visualization
    cmap_dropdown = vp.menu(choices=['jet', 'viridis', 'plasma', 'inferno', 'magma'], bind=cmap_update)
    geom.scene.append_to_caption("""\n\n""")
    geom.scene.append_to_caption("""strain scale: """)
    slider1 = vp.slider(min=0, max=strain_scale, value=20, bind=slider_strain_scale, textwrap=True, length=400)
    geom.scene.append_to_caption("""\n\n""")
    geom.scene.append_to_caption("""time scale: """)
    slider2 = vp.slider(min=0, max=10, value=1, bind=slider_time_scale, textwrap=True, length=400)

    text = vp.wtext(text='fps: ')

    # Simulation player loop
    while True:
        for sol_t in range(len(x_sol.t)):
            while paused:
                time.sleep(0.05)
            t0 = time.time()
            y = x_sol.y[:, sol_t]
            geom.nodes = y[:i_max * j_max * 3].reshape((i_max, j_max, 3))
            time_slider.value = x_sol.t[sol_t]

            if not show_strain:
                geom.update_vp_nodes([[vp.color.white for j in range(0, j_max)] for i in range(0, i_max)])
            else:
                strains = [[0 for j in range(0, j_max)] for i in range(0, i_max)]
                colors = [[vp.color.white for j in range(0, j_max)] for i in range(0, i_max)]

                for bar in geom.bars:
                    [p0, p1, l0] = bar
                    strain = abs(np.linalg.norm(geom.nodes[p0[0]][p0[1]] - geom.nodes[p1[0]][p1[1]]) / l0 - 1)
                    strains[p0[0]][p0[1]] += strain * strain_scale
                    strains[p1[0]][p1[1]] += strain * strain_scale

                for i in range(0, i_max):
                    for j in range(0, j_max):
                        colors[i][j] = interpolate_color(strains[i][j], cmap)
                        # give color according to nodal position
                        # colors[i][j] = interpolate_color((i * i_max + j) / (i_max * j_max), cmap)
                geom.update_vp_nodes(colors=colors)

            while time.time() - t0 < dt * time_scale:
                pass
            text.text = 'fps: ' + str(int(1 / (time.time() - t0)))
