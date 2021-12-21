from agent.controllers.analytical_agent import AnalyticalAgent
from agent.nn_agents.nn_grav_agent import NNGravityAgent
from plotter import Plotter
from agent.controllers.gcpd_agent import GCPDAgent
from agent.nn_agents.nn_agent import NNAgent
from agent.nn_agents.nn_nop_agent import NopAgent
from agent.nn_agents.nn_testing_agent import NNTestingAgent
from settings.ExecutionMode import ExecutionMode
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.FileHandler import FileHandler
from utils.data_transformation import *
from exceptions.CollisionException import CollisionException
from log.info_str import get_info_str
from settings.settings_access import settings
import time
from agent.AgentType import AgentType
from pathlib import Path
from utils.progresbar import resetBar

agent_type = {
    AgentType.ANALYTICAL.value: lambda target_pos, _          : AnalyticalAgent(target_pos),
    AgentType.GCPD.value:       lambda target_pos, _          : GCPDAgent(target_pos),
    AgentType.NN.value:         lambda target_pos, model_path : NNAgent(target_pos, model_path),
    AgentType.NN_TESTING.value: lambda target_pos, model_path : NNTestingAgent(target_pos, model_path),
    AgentType.NN_NOP.value:     lambda target_pos, model_path : NopAgent(target_pos, model_path),
    AgentType.NN_GRAV.value:    lambda target_pos, model_path : NNGravityAgent(target_pos, model_path),
}

def check_collision(agent, particles, intial_agent_mass, is_testing):    
    does_collide = particles[settings.agent_index].m > intial_agent_mass

    if is_testing :
        agent.data_storage["collision"] = does_collide

    if does_collide:
        raise CollisionException("A collision involving the agent has happened.", agent)

def run(archive_fname, model_path, is_testing, config=None):
    if config is None :
        config = gen_valid_environment()

    target_pos = config["target_pos"]
    particles = config["particles"]

    agent = agent_type[settings.agent_type](target_pos, model_path)

    sim = setup(agent, archive_fname, particle_list=particles)
    sim.integrate(settings.sim_time)
    
    check_collision(agent, sim.particles, particles[settings.agent_index]['mass'], is_testing)

    archive = rebound.SimulationArchive(archive_fname)
    return target_pos, archive, agent

def handle_run(archive_fname, model_path, is_testing=False, config = None):
    run_succeeded = True
    run_data = None
    try:
        start_time = time.time()
        run_data = run(archive_fname, model_path, is_testing, config)
        time_spent = f"Time spent: {time.time() - start_time} seconds"
    except CollisionException as e:
        run_succeeded = False
        error_message = e.args[0]
        run_data = (None, None, e.agent)

    status = {
        'run_succeeded': run_succeeded,
        'time_spent': time_spent if run_succeeded else None,
        'error_message': error_message if not run_succeeded else None
    }
    return run_data, status

def do_testing_run():

    data_dir = FileHandler.get_data_dir("saved_models")
    env_path = FileHandler.get_data_dir("environments")

    environments = FileHandler.read_json(Path.joinpath(env_path, "environments.json"))

    data = FileHandler.get_data_files(data_dir)

    for i in range(settings.num_of_iterations):
        for model in data:
            model_folders = FileHandler.get_data_files(Path.joinpath(data_dir, model))
      
            folder = Path.joinpath(FileHandler.get_data_dir("testing_data"), model)
            FileHandler.ensure_dir_exists(folder)

            archive_path = f"{folder}/archive_{i}"
            
            model_path = f"saved_models/{model}/{[dir for dir in model_folders if not 'history' in dir ][0]}"
            run_data, status = handle_run(archive_path + ".bin", model_path, config=environments[i])

            _, _, agent = run_data

            FileHandler.write_json(archive_path + ".json", agent.data_storage)

            info_str = get_info_str(i, status)
            print(f"\n{model}: {info_str}\n")
            resetBar()

def do_normal_run():
    successful_runs = 0
    for i in range(settings.num_of_iterations):
        file_handler = FileHandler(settings.agent_type)
        archive_fname = file_handler.get_default_file_path(settings.bin_file_ext)
        run_data, status = handle_run(archive_fname, settings.nn_model_path)
        info_str = get_info_str(i, status)

        print(f"\n{info_str}\n")

        if status['run_succeeded']:
            successful_runs += 1
            (target_pos, archive, agent) = run_data
            if  (settings.write_data_to_files and
                (agent_type == AgentType.ANALYTICAL.value or agent_type == AgentType.GCPD.value)):
                archive_as_json = get_archive_as_json_str(archive, agent, target_pos)
                file_handler.write_to_file(settings.json_file_ext, archive_as_json, file_handler.agent_path_json)

    if (successful_runs > 0):
        plotter = Plotter()
        plotter.plot_3d(archive, target_pos)
        plotter.show_plots()

def do_infinite_run():
    successful_runs = 0
    run_count = 0
    batch = []
    while True:
        file_handler = FileHandler(settings.agent_type)
        archive_fname = file_handler.get_abs_path_of_file(settings.bin_file_ext)
        run_data, status = handle_run(archive_fname, settings.nn_model_path)
        
        print(get_info_str(run_count, status))
        run_count += 1

        if status['run_succeeded']:
            successful_runs += 1
            batch.append(run_data)
            if len(batch) >= settings.batch_size and settings.write_data_to_files:
                archive_as_json = get_batch_as_json_str(batch)
                file_handler.write_to_file(settings.json_file_ext, archive_as_json)
                batch = []

execution_mode = {
    ExecutionMode.NORMAL.value: do_normal_run,
    ExecutionMode.TESTING.value: do_testing_run,
    ExecutionMode.INFINITE.value: do_infinite_run
}

if __name__ == "__main__":
    execution_mode[settings.execution_mode]()
