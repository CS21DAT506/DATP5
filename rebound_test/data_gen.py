from agent.controllers.gcpd_agent import GCPDAgent
from settings.settings_access import settings
from sim_setup.bodies import *
from utils.FileHandler import FileHandler
import json
from agent import *

def get_agent_gravity(agent_pos, planets, G=6.646596924499661e-05):
    agent_acc = np.array((0, 0))
    agent_pos = np.array(agent_pos)

    for planet in planets:
        distance = np.array(planet["pos"][:2]) - agent_pos
        agent_acc = agent_acc + planet["mass"] * distance / np.linalg.norm(distance)**3

    return agent_acc * G

def gen_unsimulated_gcpd_data(save_path):
    """Generates data from GCPD without simulating in Rebound"""
    input_data_array = []
    ouput_data_array = []

    for i in range(settings.num_of_iterations):
        print(f"Iteration {i}")
        agent = get_agent(use_random_pos=True)
        gcpd_agent = GCPDAgent(target_pos)

        bodies, target_pos = get_environment(settings.num_of_planets)
        grav_vector = np.array([0, 0, 0]) if settings.num_of_planets <= 0 else get_agent_gravity(agent['pos'][:2], bodies[1:])

        input_data_array.append([ *target_pos[:2], *agent['pos'][:2], *agent['vel'][:2] ])

        acc = gcpd_agent._get_agent_acceleration(agent['pos'], agent['vel'], grav_vector)
        ouput_data_array.append([ *acc[:2] ]) #TODO Please check <----------------------------------------------------------------------

    data_points = { 'input': input_data_array, 'output': ouput_data_array }
    
    FileHandler.write_json(save_path, data_points)

def environment_gen():

    environments = []
    num_of_environments = 10000
    for i in range(num_of_environments):
        environments.append(get_valid_environment())
        if i % 100 == 0:
            print(str(i / 100) + " %")

    data_dir = FileHandler.get_data_dir("rebound_test/environments")

    print("Environments generated!")
    FileHandler.write_json(str(data_dir) + "/environments.json", environments)

if __name__ == "__main__":
    ...