from numpy import linalg
from agent.agent_base import AgentBase
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from settings.settings_access import settings 

class NNAgent(AgentBase):

    def __init__(self, target_pos, save_path):
        super().__init__(target_pos)
        self.model = keras.models.load_model(save_path, compile=False)
        self.time = -1
        self.output = np.array([])

    def expand_vector_dim(self, vec):
        return np.append(vec, [0])

    def _get_agent_acceleration(self, sim):
        nn_input_data = [self.target_pos[0], self.target_pos[1]]

        for particle in sim.particles:
            nn_input_data.extend([particle.x, particle.y, particle.vx, particle.vy, particle.m])

        return self.expand_vector_dim(self.model([nn_input_data])[0]) 

    def get_thrust(self, sim):

        if self.time != sim.t:            
            self.time = sim.t
            self.output = self._get_agent_acceleration(sim) 

            if np.linalg.norm(self.output) > settings.max_acceleration:
                self.output = self.output *  settings.max_acceleration / np.linalg.norm(self.output)

        return self.output
       
        
