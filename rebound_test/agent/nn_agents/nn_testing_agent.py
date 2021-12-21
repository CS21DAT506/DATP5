import numpy as np
from agent.controllers.gcpd_agent import GCPDAgent
from agent.nn_agents.nn_agent import NNAgent
from math import floor
from agent.nn_agents.nn_grav_agent import NNGravityAgent
from utils.progresbar import UPDATE_CONST, bar
import time

class NNTestingAgent(NNGravityAgent):
    def __init__(self, target_pos, save_path):
        super().__init__(target_pos[:2], save_path)
        self.data_storage =    {"agent_acceleration": [], 
                                "gcpd_acceleration": [], 
                                "dist_to_target": [], 
                                "agent_time": [], 
                                "gcpd_time": [],
                                "overhead_time": [],
                                "grav_acceleration": []}
        self.gcpd = GCPDAgent(target_pos)

    def _get_agent_acceleration(self, sim):
        start_time = time.time()

        particle = sim.particles[0]

        agent_pos = np.array( (particle.x, particle.y, 0) )
        grav = self._get_agent_gravity(agent_pos, sim )
        nn_input_data = [self.target_pos[0], self.target_pos[1], particle.x, particle.y, particle.vx, particle.vy, grav[0], grav[1]]

        overhead_time = time.time() - start_time
        res = self.expand_vector_dim(self.model(np.array([nn_input_data])))
        agent_time = time.time() - start_time - overhead_time

        gcpd_res = self.gcpd.get_agent_acceleration(agent_pos, np.array((particle.vx, particle.vy, 0)), grav)
        gcpd_time = time.time() - start_time - agent_time - overhead_time

        if sim.t * UPDATE_CONST - floor(sim.t * UPDATE_CONST) < 0.01 * UPDATE_CONST:
            bar.next()

        self.data_storage["agent_acceleration"].append([*res])
        self.data_storage["gcpd_acceleration"].append([*gcpd_res])
        self.data_storage["dist_to_target"].append(np.linalg.norm(self.target_pos - np.array(agent_pos[:2])))
        self.data_storage["agent_time"].append(agent_time)
        self.data_storage["gcpd_time"].append(gcpd_time)
        self.data_storage["overhead_time"].append(overhead_time)
        self.data_storage["grav_acceleration"].append([*grav])

        return res
