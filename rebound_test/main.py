import rebound
import random
import math

from constants import *
from analytical_agent import AnalyticalAgent
from plotter import Plotter

from setup.setup import *
from setup.bodies import *


def main():
    particles = [
        agent,
        planets[0]
    ]

    is_valid_conf = is_valid_configuration(agent, planets, target_pos, MIN_DIST_TO_TARGET)
    if (not is_valid_conf):
        raise Exception("Invalid configuration.")

    analytical_agent = AnalyticalAgent(target_pos)
    sim = setup(analytical_agent, particle_list=particles)

    particle_plot = [[] for _ in sim.particles]    
    time = []

    for i in range(SIM_TIME*10):

        for j in [i for i in range(len(sim.particles))]:
            particle_plot[j].append((sim.particles[j].x, sim.particles[j].y))
        time.append(sim.t)
        sim.integrate(i*0.1)

    plotter = Plotter()
    # plotter.plot_2d(particle_plot, sim)
    plotter.plot_3d(particle_plot, time, target_pos)
    plotter.show_plots()

if __name__ == "__main__":
    current_iteration = 1
    for i in range(NUM_OF_ITERATIONS):
        print(f"Iteration: {i}.")
        main()

