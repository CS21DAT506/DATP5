from agent.gcpd_agent import GCPDAgent
from settings.SettingsAccess import settings
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

if __name__ == "__main__":
    input_data_array = []
    ouput_data_array = []

    for i in range(settings.num_of_iterations):
        print(f"Iteration {i}")
        agent = get_agent(use_random_pos=True)
        target_pos = get_target_pos()
        gcpd_agent = GCPDAgent(target_pos)

        bodies = get_particles(settings.num_of_planets)

        grav_vector = get_agent_gravity(agent['pos'][:2], bodies[1:])

        input_data_array.append( [ *target_pos[:2], *agent['pos'][:2], *agent['vel'][:2] ] )

        acc = gcpd_agent._get_agent_acceleration(agent['pos'], agent['vel'], grav_vector)
        ouput_data_array.append( [ *acc[:2] ] )

    data_points = { 'input': input_data_array, 'output': ouput_data_array }
    json_str = json.dumps( data_points, indent=4 )
    
    with open("validation_data.json", "w") as file:
                file.write(json_str)

    # for _ in range(settings.num_of_iterations):
    #     bodies = get_particles(settings.num_of_planets)
    #     target_pos = get_target_pos()
    #     agent = GCPDAgent(target_pos)

    #     agent_pos = bodies[0]

    #     grav_vector = agent._get_agent_gravity(agent_pos, bodies[1:])
        
    #     input = [target_pos[0], target_pos[1], agent_pos[0], agent_pos[1], ]
    # ...