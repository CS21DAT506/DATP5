from numpy import linalg
from agent.agent_base import AgentBase
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from settings.SettingsAccess import settings 

class NNAgent(AgentBase):

    def __init__(self, target_pos, save_path):
        super().__init__(target_pos)
        self.model = keras.models.load_model(save_path, compile=False)
        # tf.compat.v1.disable_eager_execution()
        self.time = -1
        self.output = np.array([])

    def _get_agent_acceleration(self, sim):
        nn_input_data = [self.target_pos[0], self.target_pos[1]]

        for particle in sim.particles:
            nn_input_data.extend([particle.x, particle.y, particle.vx, particle.vy, particle.m])

        # start_t = time.time()
        res = np.append( self.model.predict([nn_input_data])[0], [0] ) # add 0 as the z-axis
        # print(f"Finished predicting. Time spent: {time.time() - start_t}")
        return res 

    def get_thrust(self, sim):
        # print(f"Get thrust. sim.t: {sim.t}")

        if self.time != sim.t:            
            self.time = sim.t
            self.output = self._get_agent_acceleration(sim) 

            if np.linalg.norm(self.output) > settings.max_acceleration:
                self.output = self.output *  settings.max_acceleration / np.linalg.norm(self.output)

        return self.output
       
        
