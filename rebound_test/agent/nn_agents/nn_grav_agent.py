import numpy as np
from P5.rebound_test.agent.gcpd_agent import GCPDAgent
from agent.nn_agent import NNAgent
from math import floor
from utils.progresbar import UPDATE_CONST, bar
import time

class NNGravityAgent(NNAgent):
    def __init__(self, target_pos, save_path, data_storage = None):
        super().__init__(target_pos[:2], save_path, data_storage)
        self.gcpd = GCPDAgent(self.target_pos[:2])

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
        grav = self._get_agent_gravity(agent_pos, sim )
        nn_input_data.extend([grav[0], grav[1]])

        start_time = time.time()
        res = np.append( self.model.predict([nn_input_data])[0], [0] ) # add 0 as the z-axis
        agent_time = start_time - time.time()
        gcpd_res = self.gcpd.get_agent_acceleration(agent_pos, np.arrar((particle.vx, particle.vy)), grav)
        gcpd_time = start_time - time.time() - agent_time

        if sim.t * UPDATE_CONST - floor(sim.t * UPDATE_CONST) < 0.01 * UPDATE_CONST:
            bar.next()

        self.data_storage["agent_acceleration"].append(res)
        self.data_storage["gcpd_acceleration"].append(gcpd_res)
        self.data_storage["dist_to_target"].append(np.linalg.norm(self.target_pos - agent_pos))
        self.data_storage["agent_time"].append(agent_time)
        self.data_storage["gcpd_time"].append(gcpd_time)

        return res
