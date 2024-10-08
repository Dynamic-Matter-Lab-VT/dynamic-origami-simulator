import numpy as np
import vpython as vp
import pickle
import os

if __name__ == '__main__':
    data_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) + '/../data/nl_spring_mass_damper_raw_data/'

    # data = pickle.load(open(data_dir + 'linear_10.pkl', 'rb'))
    data = pickle.load(open(data_dir + 'pure_sine_5.pkl', 'rb'))
    y = data['y']
    t = data['t']
    n = data['n']
    m = t.shape[0]
    dt = t[1] - t[0]

    x = np.zeros((n, m))
    for i in range(n):
        x[i, :] = i + y[i, :]

    # make a vpython animation
    scene = vp.canvas(title='Spring-Mass-Damper System', width=1000, height=800,
                      background=vp.color.white, align='left')
    scene.fov = np.pi / 8
    scene.center = vp.vector(n / 2, 0, 0)

    vertices = [vp.vector(x[i, 0], 0, 0) for i in range(n)]
    spheres = [vp.sphere(pos=vertices[i], radius=0.15, color=vp.color.red) for i in range(n)]

    springs = [vp.helix(pos=vertices[i], axis=vertices[i + 1] - vertices[i], radius=0.1, color=vp.color.blue) for i in
               range(n - 1)]

    j = 0
    # start capturing the animation
    paused = True


    def pause_button_func(p):
        global paused
        paused = not paused
        if paused:
            pause_button.text = 'Play'
        else:
            pause_button.text = 'Pause'


    scene.append_to_caption('\n')
    pause_button = vp.button(text='Play', bind=pause_button_func)

    while True:
        vp.rate(1 / dt)
        if paused:
            continue
        else:
            for i in range(n):
                vertices[i].x = x[i, j]
                spheres[i].pos = vertices[i]
                if i < n - 1:
                    springs[i].pos = vertices[i]
                    springs[i].axis = vertices[i + 1] - vertices[i]

            j += 1
            if j == m:
                j = 0
                paused = True
