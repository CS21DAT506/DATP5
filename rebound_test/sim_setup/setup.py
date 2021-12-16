import rebound
import random
import math
import numpy as np

from settings.settings_access import settings

def get_vector_with_circular_bound(max_radius):
    r = math.sqrt(random.random()) * max_radius
    a = (2 * random.random() * math.pi)
    x = math.cos(a) * r
    y = math.sin(a) * r
    return np.array( [x, y, 0] )

def add_particle(sim, pos, vel, mass, radius):
    sim.add(m = mass, r = radius, x = pos[0], y = pos[1], z = pos[2], vx = vel[0], vy = vel[1], vz = vel[2])

def setup(agent, archive_fname, particle_list=[]):
    """
        sets up a sim based on either a list of particles or randomised based on amount
    """
    sim = rebound.Simulation()
    sim.integrator = "SABA(10,6,4)"
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.dt = 0.01
    sim.units = ("kg", "km", "yr")
    sim.additional_forces = agent.add_thrust
    sim.force_is_velocity_dependent = 1
    sim.automateSimulationArchive(archive_fname, interval=settings.sim_time / settings.num_of_data_points)

    for particle in particle_list:
        add_particle(sim, particle["pos"], particle["vel"], particle["mass"], particle["radius"])
    
    return sim
