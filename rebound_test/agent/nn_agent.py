from agent_base import AgentBase
import tensorflow.keras as keras

class NNAgent(AgentBase):

    def _get_agent_acceleration(self, sim):

        def __init__(self, target_pos, save_path):
            self.super().__init__(target_pos)
            self.model = keras.models.load_model(save_path)

        nn_input_data = [self.target_pos[0], self.target_pos[1]]

        for particle in sim.particles:
            nn_input_data.extend([particle.x, particle.y, particle.vx, particle.vy, particle.m])

        return self.model.predict(nn_input_data)[0] + [0] # add 0 as the z-axis


    def get_thrust(self, archive):
        sim = archive
        return self._get_agent_acceleration(sim)

        
        
