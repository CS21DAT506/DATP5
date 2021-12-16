from agent.controllers.gcpd_agent import GCPDAgent
from sim_setup.setup import *
from sim_setup.bodies import *
from utils.data_transformation import *
from settings.settings_access import settings
from pathlib import Path
from utils.FileHandler import FileHandler

def sanity_check_data():
    data = None
    sample_size = 10000

    data_dir = FileHandler.get_data_dir(settings.data_dir_name)
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

if __name__ == "__main__":
    sanity_check_data()

