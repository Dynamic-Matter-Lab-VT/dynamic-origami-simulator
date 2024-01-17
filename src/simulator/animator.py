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

show_strain = False
paused = False
strain_scale = 20
time_scale = 1


def interpolate_color(value):
    """
        Interpolates a color between blue and green for values less than or equal to 0.5,
        and between green and red for values greater than 0.5.

        Args:
            value (float): The value between 0 and 1 used to interpolate the color.

        Returns:
            vpython.vector: The interpolated color as a VPython vector.

    """
    blue = vp.vector(0, 0, 1)
    green = vp.vector(0, 1, 0)
    red = vp.vector(1, 0, 0)
    if value <= 0.5:
        ratio = 2 * value
        interpolated_color = blue + ratio * (green - blue)
    else:
        ratio = 2 * (value - 0.5)
        interpolated_color = green + ratio * (red - green)
    return interpolated_color


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


def simulate(geom_, solution_):
    x_sol = solution_
    params = x_sol.sol
    geom = geom_
    dt = params[-1]

    i_max = geom.i_max
    j_max = geom.j_max

    vp.scene.append_to_caption('\n')

    # Button to pause the simulation
    pause_button = vp.button(text='Pause', bind=pause_button_func)

    # Checkbox to toggle strain visualization
    strain_checkbox = vp.checkbox(text='Show Strain', bind=strain_checkbox_func)

    # Slider to adjust the strain scale
    slider1 = vp.slider(min=0, max=50, value=20, bind=slider_strain_scale, textwrap=True, length=400)

    # Slider to adjust the time scale
    slider2 = vp.slider(min=1, max=10, value=1, bind=slider_time_scale, textwrap=True, length=400)

    # Text display for FPS
    text = vp.wtext(text='fps: 0')

    # Simulation player loop
    while True:
        for sol_t in range(len(x_sol.t)):
            while paused:
                time.sleep(0.05)
            t0 = time.time()
            y = x_sol.y[:, sol_t]
            geom.nodes = y[:i_max * j_max * 3].reshape((i_max, j_max, 3))

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
                        colors[i][j] = interpolate_color(strains[i][j])
                geom.update_vp_nodes(colors=colors)

            while time.time() - t0 < dt * time_scale:
                pass
            text.text = 'fps: ' + str(int(1 / (time.time() - t0)))
