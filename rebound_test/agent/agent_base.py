from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self, target_pos, data_storage=None) -> None:
        self.target_pos = target_pos
        self.data_storage = data_storage

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