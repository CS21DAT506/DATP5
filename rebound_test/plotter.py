import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
import matplotlib.cm as cm


class Plotter():

    def plot_2d(self, particle_plot, sim):
        COLOR = cm.rainbow(np.linspace(0, 1, len(sim.particles)))
        plot_design = ["-o", "-d"]

        for i in range(len(particle_plot)):
            particle = particle_plot[i]

            x = [pos[0] for pos in  particle]
            y = [pos[1] for pos in  particle]

            plt.plot(x, y, color=COLOR[i])


    def plot_3d(self, particle_plot, time):
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        axe = fig.gca(projection='3d')

        for i in range(len(particle_plot)):
            particle = particle_plot[i]

            x = [pos[0] for pos in  particle]
            y = [pos[1] for pos in  particle]

            # if(i == 0):
            #     axe.set_box_aspect( (np.ptp(x), np.ptp(y), np.ptp(time)) )
            obj_label = "planet" if i > 0 else "agent"
            axe.plot(x, y, time, "o", label=obj_label)

        axe.legend()

    def show_plots(self):
        plt.show()
