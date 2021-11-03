import numpy as np
from agent.nn_agent import NNAgent
from settings.SettingsAccess import settings 
from math import floor

from progress.bar import IncrementalBar
bar = IncrementalBar('Elapsed time', max=settings.sim_time)

class NopAgent(NNAgent):

    def _get_agent_acceleration(self, sim):
        nn_input_data = [self.target_pos[0], self.target_pos[1]]

        particle = sim.particles[0]
        nn_input_data.extend([particle.x, particle.y, particle.vx, particle.vy])

        # start_t = time.time()
        res = np.append( self.model.predict([nn_input_data])[0], [0] ) # add 0 as the z-axis
        # print(f"Finished predicting. Time spent: {time.time() - start_t}")
        # print(f"RES: {res}")

        if sim.t - floor(sim.t) < 0.01:
            # print(f"sim.t: {sim.t}")
            bar.next()
        return res
