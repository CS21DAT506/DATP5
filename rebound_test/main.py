import rebound
from agent.analytical_agent import AnalyticalAgent
from plotter import Plotter
from agent.gcpd_agent import GCPDAgent
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.data_saving import *
from utils.data_transformation import *
from exceptions.CollisionException import CollisionException
from utils.performance_tracker import calculate_run_performance
from log.info_str import get_info_str
from settings.SettingsAccess import settings
import time

def check_collision(particles, intial_agent_mass):
    agent = particles[settings.agent_index]
    if agent.m > intial_agent_mass:
        raise CollisionException(f"A collision involving the agent has happened.")
        
def run():
    is_valid_conf = False
    while not is_valid_conf:
        particles = get_particles(settings.num_of_planets)
        # particles = get_colliding_particles()
        target_pos = get_target_pos()
        # target_pos = np.array( (500, -500, 0) )
        is_valid_conf = is_valid_configuration(particles[settings.agent_index], particles[settings.agent_index+1:], target_pos, settings.min_dist_to_target)

    agent = GCPDAgent(target_pos)

    file_name = get_timestamp_str()
    archive_fname = get_abs_path_of_file(file_name, settings.bin_file_ext)
    sim = setup(agent, archive_fname, particle_list=particles)

    sim.integrate(settings.sim_time)
    sim.particles[0].m 
    check_collision(sim.particles, particles[settings.agent_index]['mass'])

    archive = rebound.SimulationArchive(archive_fname)
    performance = calculate_run_performance(archive, target_pos)
    archive_as_json = get_archive_as_json_str(archive, agent, target_pos)
    write_to_file(file_name, settings.json_file_ext, archive_as_json)
    return target_pos, archive

if __name__ == "__main__":
    successful_runs = 0
    run_succeeded = True
    for i in range(settings.num_of_iterations):
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
