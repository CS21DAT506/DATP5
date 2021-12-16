from agent.agent_base import AgentBase
import numpy as np
from settings.settings_access import settings
MAX_ACCELERATION = settings.max_acceleration

class GCPDAgent(AgentBase):
    _progression_const = 0.15

    def _speed_coefficient(self):
        return -np.log(1 - self._progression_const)

    def _get_velocity_change(self, agent_pos, agent_velocity):
        distance_to_target = self.target_pos - agent_pos
        return self._speed_coefficient() * distance_to_target - agent_velocity
    
    def _get_u_value(self, normalized_velocity_change, agent_gravity):
        cross_length = np.linalg.norm(np.cross(normalized_velocity_change, agent_gravity))
        if (cross_length > MAX_ACCELERATION):
            return -1

        return np.dot(normalized_velocity_change, agent_gravity) + np.sqrt(MAX_ACCELERATION**2 - cross_length**2)

    def get_agent_acceleration(self, agent_pos, agent_velocity, agent_gravity):
        normalized_velocity_change = self._normalize(self._get_velocity_change(agent_pos, agent_velocity))
        
        u = self._get_u_value(normalized_velocity_change, agent_gravity)
        if (u < 0):
            return - self._normalize(agent_gravity) * MAX_ACCELERATION
        
        return u * normalized_velocity_change - agent_gravity

    def _normalize(self, vec):
        return vec / np.linalg.norm(vec)

    def get_thrust(self, sim):
        agent = sim.particles[0]

        agent_pos = np.array( (agent.x, agent.y, agent.z) )
        agent_velocity = np.array( (agent.vx, agent.vy, agent.vz) )
        agent_gravity = self._get_agent_gravity(agent_pos, sim)

        return self._get_agent_acceleration(agent_pos, agent_velocity, agent_gravity)