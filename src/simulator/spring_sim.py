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


def update_sim_time(t):
    pass


def simulate(geom_, solution_):
    global show_strain, paused, strain_scale, time_scale, pause_button, strain_checkbox, slider1, slider2, text, curr_time, time_slider, geom
    x_sol = solution_
    params = x_sol.sol
    geom = geom_
    dt = params[-1]

    i_max = geom.i_max

    geom.scene.append_to_caption("""\n""")
    time_slider = vp.slider(min=0, max=x_sol.t[-1], value=0, bind=update_sim_time, textwrap=True, length=1000)
    geom.scene.append_to_caption("""\n""")

    pause_button = vp.button(text='Pause', bind=pause_button_func)
    geom.scene.append_to_caption("""\n\n""")
    strain_checkbox = vp.checkbox(text='Show Strain', bind=strain_checkbox_func)
    geom.scene.append_to_caption("""\n\n""")
    geom.scene.append_to_caption("""strain scale: """)
    slider1 = vp.slider(min=0, max=50000, value=20, bind=slider_strain_scale, textwrap=True, length=400)
    geom.scene.append_to_caption("""\n\n""")
    geom.scene.append_to_caption("""time scale: """)
    slider2 = vp.slider(min=0, max=10, value=1, bind=slider_time_scale, textwrap=True, length=400)

    text = vp.wtext(text='fps: ')

    while True:
        for sol_t in range(x_sol.t.size):
            while paused:
                time.sleep(0.05)
            t0 = time.time()
            geom.x = x_sol.y[:geom.i_max * 3, sol_t].reshape((geom.i_max, 3))
            curr_time = x_sol.t[sol_t]
            time_slider.value = curr_time

            if not show_strain:
                geom.update_vp_nodes([vp.color.white for j in range(0, i_max)])

            else:
                strains = [0 for j in range(0, i_max)]
                colors = [vp.color.white for j in range(0, i_max)]

                for i in range(i_max - 1):
                    strain = abs(
                        np.linalg.norm(geom.x[i] - geom.x[i + 1]) / np.linalg.norm(geom.x0[i] - geom.x0[i + 1]) - 1)
                    strains[i] += strain * strain_scale

                for i in range(i_max):
                    colors[i] = interpolate_color(strains[i])
                geom.update_vp_nodes(colors=colors)

            geom.update_spring_curves()
            while time.time() - t0 < dt * time_scale:
                pass
            text.text = 'fps: ' + str(int(1 / (time.time() - t0)))
