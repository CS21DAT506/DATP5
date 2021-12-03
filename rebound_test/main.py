from operator import is_
from agent.controllers.analytical_agent import AnalyticalAgent
from agent.nn_agents.nn_grav_agent import NNGravityAgent
from plotter import Plotter
from agent.controllers.gcpd_agent import GCPDAgent
from agent.nn_agents.nn_agent import NNAgent
from agent.nn_agents.nn_nop_agent import NopAgent
from settings.ExecutionMode import ExecutionMode
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.FileHandler import FileHandler
from utils.data_transformation import *
from exceptions.CollisionException import CollisionException
from utils.performance_tracker import calculate_run_performance
from log.info_str import get_info_str
from settings.settings_access import settings
import time
from agent.AgentType import AgentType
from pathlib import Path
import os
from utils.progresbar import resetBar

agent_type = {
    AgentType.ANALYTICAL.value: lambda target_pos, model_path : AnalyticalAgent(target_pos),
    AgentType.GCPD.value:       lambda target_pos, model_path : GCPDAgent(target_pos),
    AgentType.NN.value:         lambda target_pos, model_path : NNAgent(target_pos, model_path),
    AgentType.NN_NOP.value:     lambda target_pos, model_path : NopAgent(target_pos, model_path),
    AgentType.NN_GRAV.value:    lambda target_pos, model_path : NNGravityAgent(target_pos, model_path),
}

def check_collision(particles, intial_agent_mass, throw_exception=True):
    agent = particles[settings.agent_index]
    if agent.m > intial_agent_mass and throw_exception:
        raise CollisionException(f"A collision involving the agent has happened.")
    return agent.m > intial_agent_mass

def get_valid_environment():
    particles = None
    target_pos = None
    is_valid_conf = False
   
    while not is_valid_conf:
        particles, target_pos = get_environment(settings.num_of_planets)
        is_valid_conf = is_valid_configuration(particles[settings.agent_index], particles[settings.agent_index+1:], target_pos, settings.min_dist_to_target)
   
    return {"target_pos": target_pos, "particles": particles}

def run(archive_fname, model_path, config=None):
    is_testing = False
    if config is None :
        config = get_valid_environment()
    else:
        is_testing = True

    target_pos = config["target_pos"]
    particles = config["particles"]

    agent = agent_type[settings.agent_type](target_pos, model_path)

    sim = setup(agent, archive_fname, particle_list=particles)
    sim.integrate(settings.sim_time)
    
    collides = check_collision(sim.particles, particles[settings.agent_index]['mass'], throw_exception=not is_testing)

    if is_testing:
        agent.data_storage["collision"] = collides

    archive = rebound.SimulationArchive(archive_fname)
    return target_pos, archive, agent

def handle_run(archive_fname, model_path, config = None):
    run_succeeded = True
    run_data = None
    try:
        start_time = time.time()
        run_data = run(archive_fname, model_path, config)
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

def do_testing_run():

    data_dir = FileHandler.get_data_dir("saved_models")
    env_path = FileHandler.get_data_dir("environments")

    environments = None

    with open(Path.joinpath(env_path, "environments.json"), "r") as file:
            json_file = file.read()
            environments = json.loads(json_file)

    data = [dir for dir in FileHandler.get_data_files(data_dir)]

    for i in range(5080, settings.num_of_iterations):
        for model in data:
            model_folders = FileHandler.get_data_files(Path.joinpath(data_dir, model))

            model_path = "saved_models/" + model + "/" + [dir for dir in model_folders if "2021" in dir ][0]
      
            folder = Path.joinpath(FileHandler.get_data_dir("testing_data"), model)
            FileHandler.ensure_dir_exists(folder)

            archive_path = str(folder) + "/archive_" + str(i)
            
            archive_fname = archive_path + ".bin"
            run_data, status = handle_run(archive_fname, model_path, config=environments[i])

            target_pos, archive, agent = run_data

            storage = agent.data_storage

            with open(archive_path + ".json", "w") as file:
                jsonstr = json.dumps(storage, indent=4)
                file.write(jsonstr)

            info_str = get_info_str(i, status)
            print(f"\n{model}: {info_str}\n")
            resetBar()

def do_normal_run():
    successful_runs = 0
    for i in range(settings.num_of_iterations):
        file_handler = FileHandler(settings.agent_type)
        archive_fname = file_handler.get_abs_path_of_file(settings.bin_file_ext)

        run_data, status = handle_run(archive_fname, settings.nn_model_path)
        info_str = get_info_str(i, status)

        print(f"\n{info_str}\n")

        if status['run_succeeded']:
            successful_runs += 1 
            (target_pos, archive, agent) = run_data
            performance = calculate_run_performance(archive, target_pos)
            if (settings.write_data_to_files):
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
        run_data, status = handle_run(archive_fname, settings.nn_model_path)
        info_str = get_info_str(run_count, status)
        print(info_str)
        run_count += 1

        if status['run_succeeded']:
            successful_runs += 1 
            batch.append(run_data)
            if len(batch) >= settings.batch_size and settings.write_data_to_files:
                archive_as_json = get_batch_as_json_str(batch)
                file_handler.write_to_file(settings.json_file_ext, archive_as_json)
                batch = []

def simple_data_gen():
    input_data_array = []
    ouput_data_array = []

    for i in range(settings.num_of_iterations):
        print(f"Iteration {i}")
        agent = get_agent(use_random_pos=True)
        target_pos = get_target_pos()
        gcpd_agent = GCPDAgent(target_pos)

        input_data_array.append( [ *target_pos[:2], *agent['pos'][:2], *agent['vel'][:2] ] )
    
        acc = gcpd_agent._get_agent_acceleration(agent['pos'], agent['vel'], np.array( [0, 0, 0] ))
        ouput_data_array.append( [ *acc[:2] ] )

    data_points = { 'input': input_data_array, 'output': ouput_data_array }
    json_str = json.dumps( data_points, indent=4 )
    f_handler = FileHandler(settings.agent_type)
    f_handler.write_to_file(settings.json_file_ext, json_str)

def environment_gen():

    environments = []
    num_of_environments = 10000
    for i in range(num_of_environments):
        environments.append(get_valid_environment())
        if i % 100 == 0:
            print(str(i / 100) + " %")

    data_dir = FileHandler.get_data_dir("rebound_test/environments")

    print("Environments generated!")

    with open(str(data_dir) + "/environments.json", "w") as file:
        jsonstr = json.dumps(environments, indent=4)
        file.write(jsonstr)

    ...

def get_data_dir(dir_name): 
    return Path.joinpath(Path().resolve(), dir_name)

def get_data_files(data_dir):
    return os.listdir(data_dir)

def sanity_check_data():
    data = None
    sample_size = 10000

    data_dir = get_data_dir(settings.data_dir_name)
    path_to_json_file = Path.joinpath(data_dir, "gravity_vector_data.json")
    with open(path_to_json_file) as file:
        data = json.load(file)

    for _ in range(sample_size):
        index = random.randrange(0, len(data["input"]))
        target = np.array(data["input"][index][0:2])
        agent_pos = np.array(data["input"][index][2:4])
        agent_vel = np.array(data["input"][index][4:6])
        agent_grav = np.array(data["input"][index][6:8])
        actual_acc = np.array(data["output"][index])
        agent = GCPDAgent(target)
        agent_out = agent._get_agent_acceleration(agent_pos, agent_vel, agent_grav)
        if np.linalg.norm(actual_acc - agent_out) > 0.001:
            raise Exception("data not within accepted values")



execution_mode = {
    ExecutionMode.NORMAL.value: do_normal_run,
    ExecutionMode.TESTING.value: do_testing_run,
    ExecutionMode.INFINITE.value: do_infinite_run,
    ExecutionMode.SIMPLE_DATA_GEN.value: simple_data_gen,
    ExecutionMode.DATA_SANITY_CHECK.value: sanity_check_data,
    ExecutionMode.ENVIRONMENT_GEN.value: environment_gen
}

if __name__ == "__main__":
    execution_mode[settings.execution_mode]()
