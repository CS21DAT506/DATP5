import numpy as np
from agent.nn_agent import NNAgent
from math import floor
from utils.progresbar import UPDATE_CONST, bar

class NNGravityAgent(NNAgent):
    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array( (0, 0) )
        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array( (particle.x, particle.y) ) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        return agent_acc * sim.G

    def _get_agent_acceleration(self, sim):
        nn_input_data = [self.target_pos[0], self.target_pos[1]]

        particle = sim.particles[0]
        nn_input_data.extend([particle.x, particle.y, particle.vx, particle.vy])

        agent_pos = np.array( (particle.x, particle.y) )
        acc = self._get_agent_gravity(agent_pos, sim )
        nn_input_data.extend([acc[0], acc[1]])

        res = np.append( self.model.predict([nn_input_data])[0], [0] ) # add 0 as the z-axis

        if sim.t * UPDATE_CONST - floor(sim.t * UPDATE_CONST) < 0.01 * UPDATE_CONST:
            bar.next()
        return res
