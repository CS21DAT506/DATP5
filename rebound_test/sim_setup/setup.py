import rebound
import random
import math
import numpy as np

from settings.settings import NUM_OF_DATA_POINTS, SIM_TIME

def get_distribution(max_radius, three_dimension=True):
    return (get_spherical(max_radius) if three_dimension else get_vector_with_circular_bound(max_radius))

def get_spherical(max_radius):
    r = math.pow(random.random(), 1/3) * max_radius
    a = ( 2 * random.random() * math.pi)
    h = (1 - 2 * random.random())
    p = (math.sqrt(1 - h * h))
    x = math.cos(a) * p * r
    y = math.sin(a) * p * r
    z = h * r
    return np.array( [x, y, z] )

def get_vector_with_circular_bound(max_radius):
    r = math.sqrt(random.random()) * max_radius
    a = (2 * random.random() * math.pi)
    x = math.cos(a) * r
    y = math.sin(a) * r
    return np.array( [x, y, 0] )

def add_particle(sim, pos=None, vel=None, mass=None, radius=None, three_dimension=True):
    pos = pos if pos is not None else get_distribution(10, three_dimension)
    vel = vel if vel is not None else get_distribution(0.1, three_dimension)
    mass = mass if mass is not None else random.random() * 50 + 0.5
    radius = radius if radius is not None else mass / 10000
    sim.add(m = mass, r = radius, x = pos[0], y = pos[1], z = pos[2], vx = vel[0], vy = vel[1], vz = vel[2])

def setup(agent, archive_fname, particle_list=[], amount_of_particles=None):
    """
        sets up a sim based on either a list of particles or randomised based on amount
    """
    amount_of_particles = amount_of_particles or len(particle_list)
    sim = rebound.Simulation()
    sim.integrator = "SABA(10,6,4)"
    sim.collision = "direct"
    sim.collision_resolve = "merge"  # "hardsphere"
    sim.dt = 0.01
    sim.units = ("kg", "km", "yr")
    sim.additional_forces = agent.add_thrust
    sim.force_is_velocity_dependent = 1
    sim.automateSimulationArchive(archive_fname, interval=SIM_TIME / NUM_OF_DATA_POINTS)

    for i in range(amount_of_particles):
        if i <= len(particle_list)-1:
            add_particle(sim,
                        particle_list[i]["pos"],
                        particle_list[i]["vel"],
                        particle_list[i]["mass"],
                        particle_list[i]["radius"]
            )
        else:
            add_particle(sim, three_dimension=False)
    
    return sim
