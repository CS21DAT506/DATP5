import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import matplotlib.cm as cm

def get_time(archive):
    time = []
    for a in archive:
        time.append(a.t)
    return time

class Plotter():

    def plot_2d(self, particle_plot, sim):
        COLOR = cm.rainbow(np.linspace(0, 1, len(sim.particles)))
        plot_design = ["-o", "-d"]

        for i in range(len(particle_plot)):
            particle = particle_plot[i]

            x = [pos[0] for pos in  particle]
            y = [pos[1] for pos in  particle]

            plt.plot(x, y, color=COLOR[i])


    def plot_3d(self, particle_plot, time, target_pos):
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        axe = fig.gca(projection='3d')

        for i in range(len(particle_plot)):
            particle = particle_plot[i]

            x = [pos[0] for pos in  particle]
            y = [pos[1] for pos in  particle]

            # if(i == 0):
            #     axe.set_box_aspect( (np.ptp(x), np.ptp(y), np.ptp(time)) )
            obj_label = "Planet" if i > 0 else "Agent"
            axe.plot(x, y, time[:len(x)], "o", label=obj_label)

        axe.plot([ target_pos[0] for _ in range(len(time)) ], [ target_pos[1] for _ in range(len(time)) ], time, "o", label="Target")

        axe.legend()

    def plot_3d(self, archive, target_pos):
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        axe = fig.gca(projection='3d')

        time = get_time(archive)

        for i in range(len(archive[0].particles)):
            x = []
            y = []
            for a in archive:
                particle = a.particles[i]

                x.append(particle.x)
                y.append(particle.y)

            # if(i == 0):
            #     axe.set_box_aspect( (np.ptp(x), np.ptp(y), np.ptp(time)) )
            obj_label = "Planet" if i > 0 else "Agent"
            axe.plot(x, y, time[:len(x)], "o", label=obj_label)

        axe.plot([ target_pos[0] for _ in range(len(time)) ], [ target_pos[1] for _ in range(len(time)) ], time, "o", label="Target")

        axe.legend()

    def show_plots(self):
        plt.show()

