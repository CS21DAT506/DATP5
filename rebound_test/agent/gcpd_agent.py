from agent.agent_base import AgentBase
import numpy as np
import time
from settings.settings_access import settings
MAX_ACCELERATION = settings.max_acceleration

class GCPDAgent(AgentBase):
    _progression_const = 0.15

    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array( (0, 0, 0) )

        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array( (particle.x, particle.y, particle.z) ) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        
        return agent_acc * sim.G

    def _speed_coefficient(self):
        return -np.log(1 - self._progression_const)

    def _get_velocity_change(self, agent_pos, agent_velocity):
        distance_to_target = self.target_pos - agent_pos
        return self._speed_coefficient() * distance_to_target - agent_velocity

    def _get_difference(self, normalized_velocity_change, agent_gravity):
        cross_length = np.linalg.norm(np.cross(normalized_velocity_change, agent_gravity))
        if (cross_length > MAX_ACCELERATION):
            return -1
        return np.sqrt(MAX_ACCELERATION**2 - cross_length**2)
    
    def _get_inverted_time(self, normalized_velocity_change, agent_gravity):
        s = self._get_difference(normalized_velocity_change, agent_gravity)
        if (s < 0):
            return s
        return np.dot(normalized_velocity_change, agent_gravity) + s

    def _get_agent_acceleration(self, agent_pos, agent_velocity, agent_gravity):
        velocity_change = self._get_velocity_change(agent_pos, agent_velocity)
        normalized_velocity_change = velocity_change / np.linalg.norm(velocity_change)
        inverted_time = self._get_inverted_time(normalized_velocity_change, agent_gravity)
        
        if (inverted_time < 0):
            return -agent_gravity / np.linalg.norm(agent_gravity) * MAX_ACCELERATION
        
        return inverted_time * normalized_velocity_change - agent_gravity

    def get_thrust(self, sim):
        print(f"Get thrust. sim.t: {sim.t}")
        agent = sim.particles[0]

        agent_pos = np.array( (agent.x, agent.y, agent.z) )
        agent_velocity = np.array( (agent.vx, agent.vy, agent.vz) )
        agent_gravity = self._get_agent_gravity(agent_pos, sim)

        start_t = time.time()
        acc = self._get_agent_acceleration(agent_pos, agent_velocity, agent_gravity)
        print(f"Finished getting acceleration. Time spent: {time.time() - start_t}")
        return acc