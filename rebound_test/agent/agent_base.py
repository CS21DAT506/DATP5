from abc import ABC, abstractmethod
import numpy as np

class AgentBase(ABC):
    def __init__(self, target_pos):
        self.target_pos = target_pos

    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array( (0, 0, 0) )

        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array( (particle.x, particle.y, particle.z) ) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        
        return agent_acc * sim.G

    @abstractmethod
    def get_thrust(self, archive):
        ...
    
    def add_thrust(self, simulation):
        sim = simulation.contents
        agent = sim.particles[0]

        agent_acc = self.get_thrust(sim)

        agent.ax += agent_acc[0]
        agent.ay += agent_acc[1]
        agent.az += agent_acc[2]