import numpy as np
from agent.nn_agents.nn_agent import NNAgent

class NNGravityAgent(NNAgent):
    def __init__(self, target_pos, save_path):
        super().__init__(target_pos[:2], save_path)

    def _get_agent_acceleration(self, sim):
        particle = sim.particles[0]

        agent_pos = np.array( (particle.x, particle.y) )
        grav = self._get_agent_gravity(agent_pos, sim )
        nn_input_data = [self.target_pos[0], self.target_pos[1], particle.x, particle.y, particle.vx, particle.vy, grav[0], grav[1]]

        return self.expand_vector_dim(self.model(np.array([nn_input_data])))
