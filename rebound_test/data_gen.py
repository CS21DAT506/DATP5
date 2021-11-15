from agent.gcpd_agent import GCPDAgent
from settings.SettingsAccess import settings
from sim_setup.bodies import *
from agent import *

def get_agent_gravity(agent_pos, planets, G=6.646596924499661e-05):
    agent_acc = np.array((0, 0))
    agent_pos = np.array(agent_pos)

    for planet in planets:
        distance = np.array(planet[:2]) - agent_pos
        agent_acc = agent_acc + planet[-1] * distance / np.linalg.norm(distance)**3

    return agent_acc * G


if __name__ == "__main__":

    for _ in range(settings.num_of_iterations):
        bodies = get_particles(settings.num_of_planets)
        target_pos = get_target_pos()
        agent = GCPDAgent(target_pos)

        agent_pos = bodies[0]

        grav_vector = agent._get_agent_gravity(agent_pos, bodies[1:])
        
        input = [target_pos[0], target_pos[1], agent_pos[0], agent_pos[1], ]
    ...