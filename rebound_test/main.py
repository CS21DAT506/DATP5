import rebound
from settings.settings import *
from analytical_agent import AnalyticalAgent
from plotter import Plotter
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.data_saving import *
from utils.data_transformation import *
from constants import *
from exceptions.CollisionException import CollisionException
import time

def check_collision(particles, intial_agent_mass):
    agent = particles[AGENT_INDEX]
    if agent.m > intial_agent_mass:
        # print(f"A collision has happened.")
        raise CollisionException(f"A collision involving the agent has happened.")

def get_info_str(iteration_num, status):
    info_str = f"Iteration: {iteration_num}{SEPARATOR} Succeeded: {status['run_succeeded']}"
    if (run_succeeded):
        info_str += f"{SEPARATOR} {status['time_spent']}"
    else:
        info_str += f"{SEPARATOR} Error message: '{status['error_message']}'"
    info_str += f"{SEPARATOR} Successful runs: {status['successful_runs']}{SEPARATOR} Total runs: {NUM_OF_ITERATIONS}"

    return info_str
        
def run():
    is_valid_conf = False
    while not is_valid_conf:
        particles = get_particles(NUM_OF_PLANETS)
        # particles = get_colliding_particles()
        target_pos = get_target_pos()
        # target_pos = np.array( (500, -500, 0) )
        is_valid_conf = is_valid_configuration(particles[AGENT_INDEX], particles[AGENT_INDEX+1:], target_pos, MIN_DIST_TO_TARGET)

    analytical_agent = AnalyticalAgent(target_pos)

    file_name = get_timestamp_str()
    archive_fname = get_abs_path_of_file(file_name, BIN_FILE_EXT)
    sim = setup(analytical_agent, archive_fname, particle_list=particles)

    sim.integrate(SIM_TIME)
    sim.particles[0].m 
    check_collision(sim.particles, particles[AGENT_INDEX]['mass'])

    archive = rebound.SimulationArchive(archive_fname)
    archive_as_json = get_archive_as_json_str(archive, analytical_agent, target_pos)
    write_to_file(file_name, JSON_FILE_EXT, archive_as_json)
    return target_pos, archive

if __name__ == "__main__":
    successful_runs = 0
    run_succeeded = True
    for i in range(NUM_OF_ITERATIONS):
        try:
            start_time = time.time()
            target_pos, archive_data = run()
            time_spent = f"Time spent: {time.time() - start_time} seconds"
            successful_runs += 1
        except CollisionException as e:
            run_succeeded = False
            error_message = e.args[0]

        status = {
            'successful_runs': successful_runs,
            'run_succeeded': run_succeeded,
            'time_spent': time_spent if run_succeeded else None,
            'error_message': error_message if not run_succeeded else None
        }

        info_str = get_info_str(i, status)


        print(info_str)

    if (successful_runs > 0):
        plotter = Plotter()
        # plotter.plot_2d(particle_plot, sim)
        plotter.plot_3d(archive_data, target_pos)
        plotter.show_plots()
