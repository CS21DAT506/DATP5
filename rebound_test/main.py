import rebound
from settings.settings import *
from analytical_agent import AnalyticalAgent
from plotter import Plotter
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.data_saving import *

def run():
    particles = [
        agent,
        planets[0]
    ]

    is_valid_conf = is_valid_configuration(agent, planets, target_pos, MIN_DIST_TO_TARGET)
    if (not is_valid_conf):
        raise Exception("Invalid configuration.")

    analytical_agent = AnalyticalAgent(target_pos)

    archive_fname = get_data_path_with_file()
    sim = setup(analytical_agent, archive_fname, particle_list=particles)

    sim.integrate(SIM_TIME)

    archive = rebound.SimulationArchive(archive_fname)
    return archive

if __name__ == "__main__":
    current_iteration = 1
    for i in range(NUM_OF_ITERATIONS):
        print(f"Iteration: {i}.")
        archive_data = run()



    plotter = Plotter()
    # plotter.plot_2d(particle_plot, sim)
    plotter.plot_3d(archive_data, target_pos)
    plotter.show_plots()
