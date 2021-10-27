from agent.analytical_agent import AnalyticalAgent
from plotter import Plotter
from agent.gcpd_agent import GCPDAgent
from rebound_test.agent.nn_agent import NNAgent
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.FileHandler import FileHandler
from utils.data_transformation import *
from exceptions.CollisionException import CollisionException
from utils.performance_tracker import calculate_run_performance
from log.info_str import get_info_str
from settings.SettingsAccess import settings
import time

from agent.AgentType import AgentType

agent_type = {
    AgentType.ANALYTICAL.value: lambda target_pos : AnalyticalAgent(target_pos),
    AgentType.GCPD.value: lambda target_pos : GCPDAgent(target_pos),
    AgentType.NN.value: lambda target_pos: NNAgent(target_pos, settings.nn_model_path)
}

def check_collision(particles, intial_agent_mass):
    agent = particles[settings.agent_index]
    if agent.m > intial_agent_mass:
        raise CollisionException(f"A collision involving the agent has happened.")

def run(archive_fname):
    is_valid_conf = False
    while not is_valid_conf:
        particles = get_particles(settings.num_of_planets)
        # particles = get_colliding_particles()
        target_pos = get_target_pos()
        # target_pos = np.array( (500, -500, 0) )
        is_valid_conf = is_valid_configuration(particles[settings.agent_index], particles[settings.agent_index+1:], target_pos, settings.min_dist_to_target)

    agent = agent_type[settings.agent_type](target_pos)

    sim = setup(agent, archive_fname, particle_list=particles)
    sim.integrate(settings.sim_time)
    check_collision(sim.particles, particles[settings.agent_index]['mass'])

    archive = rebound.SimulationArchive(archive_fname)
    return target_pos, archive, agent

def handle_run(archive_fname):
    run_succeeded = True
    run_data = None
    try:
        start_time = time.time()
        run_data = run(archive_fname)
        time_spent = f"Time spent: {time.time() - start_time} seconds"
    except CollisionException as e:
        run_succeeded = False
        error_message = e.args[0]

    status = {
        'run_succeeded': run_succeeded,
        'time_spent': time_spent if run_succeeded else None,
        'error_message': error_message if not run_succeeded else None
    }
    return run_data, status

def do_normal_run():
    successful_runs = 0
    for i in range(settings.num_of_iterations):
        file_handler = FileHandler(settings.agent_type)
        archive_fname = file_handler.get_abs_path_of_file(settings.bin_file_ext)
        run_data, status = handle_run(archive_fname)
        info_str = get_info_str(i, status)
        print(info_str)
        if status['run_succeeded']:
            successful_runs += 1 
            (target_pos, archive, agent) = run_data
            performance = calculate_run_performance(archive, target_pos)
            archive_as_json = get_archive_as_json_str(archive, agent, target_pos)
            file_handler.write_to_file(settings.json_file_ext, archive_as_json)

    if (successful_runs > 0):
        plotter = Plotter()
        # plotter.plot_2d(particle_plot, sim)
        plotter.plot_3d(archive, target_pos)
        plotter.show_plots()

def do_infinite_run():
    successful_runs = 0
    run_count = 0
    batch = []
    while True:
        file_handler = FileHandler(settings.agent_type)
        archive_fname = file_handler.get_abs_path_of_file(settings.bin_file_ext)
        run_data, status = handle_run(archive_fname)
        info_str = get_info_str(run_count, status)
        print(info_str)
        run_count += 1

        if status['run_succeeded']:
            successful_runs += 1 
            batch.append(run_data)
            if len(batch) >= settings.batch_size:
                archive_as_json = get_batch_as_json_str(batch)
                file_handler.write_to_file(settings.json_file_ext, archive_as_json)
                batch = []

if __name__ == "__main__":
    if settings.do_infinite_run:
        do_infinite_run()
    else:
        do_normal_run()
