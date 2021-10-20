import rebound
from settings.settings import *
from analytical_agent import AnalyticalAgent
from plotter import Plotter
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.data_saving import *
from utils.data_transformation import *
from constants import *
import time

def run():
    is_valid_conf = False
    while (not is_valid_conf):
        particles = get_particles(NUM_OF_PLANETS)
        target_pos = get_target_pos()
        is_valid_conf = is_valid_configuration(particles[AGENT_INDEX], particles[AGENT_INDEX+1:], target_pos, MIN_DIST_TO_TARGET)

    analytical_agent = AnalyticalAgent(target_pos)

    file_name = get_timestamp_str()
    archive_fname = get_abs_path_of_file(file_name, BIN_FILE_EXT)
    sim = setup(analytical_agent, archive_fname, particle_list=particles)

    sim.integrate(SIM_TIME)

    archive = rebound.SimulationArchive(archive_fname)
    archive_as_json = get_archive_as_json_str(archive, analytical_agent, target_pos)
    write_to_file(file_name, JSON_FILE_EXT, archive_as_json)
    return target_pos, archive

if __name__ == "__main__":
    for i in range(NUM_OF_ITERATIONS):
        start_time = time.time()
        target_pos, archive_data = run()
        print(f"Iteration: {i}, Time spent: {time.time() - start_time} seconds)")

    plotter = Plotter()
    # plotter.plot_2d(particle_plot, sim)
    plotter.plot_3d(archive_data, target_pos)
    plotter.show_plots()
