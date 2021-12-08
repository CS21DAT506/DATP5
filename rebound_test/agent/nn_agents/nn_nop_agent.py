import numpy as np
from agent.nn_agents.nn_agent import NNAgent
from settings.settings_access import settings 
from math import floor

from progress.bar import IncrementalBar
UPDATE_CONST = 4
bar = IncrementalBar('Elapsed time', max=settings.sim_time*UPDATE_CONST, suffix='%(percent)d%%')

class NopAgent(NNAgent):

    def _get_agent_acceleration(self, sim):
        particle = sim.particles[0]
        
        nn_input_data = [self.target_pos[0], self.target_pos[1], particle.x, particle.y, particle.vx, particle.vy]

        res = self.expand_vector_dim(self.model([nn_input_data])[0])

        if sim.t * UPDATE_CONST - floor(sim.t * UPDATE_CONST) < 0.01 * UPDATE_CONST:
            bar.next()
        return res
