from numpy import linalg
import rebound
import random
import math

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from plotter import Plotter

def get_distribution(max_radius, three_dimension=True):
    return (get_spherical(max_radius) if three_dimension else (*get_circular(max_radius), 0))

def get_spherical(max_radius):
    r = math.pow(random.random(), 1/3) * max_radius
    a = ( 2 * random.random() * math.pi)
    h = (1 - 2 * random.random())
    p = (math.sqrt(1 - h * h))
    x = math.cos(a) * p * r
    y = math.sin(a) * p * r
    z = h * r
    return x, y, z

def get_circular(max_radius):
    r = math.sqrt(random.random()) * max_radius
    a = (2 * random.random() * math.pi)
    x = math.cos(a) * r
    y = math.sin(a) * r
    return x, y

def add_particle(sim, pos=None, vel=None, mass=None, radius=None, three_dimension=True):
    pos = pos or get_distribution(10, three_dimension)
    vel = vel or get_distribution(0.1, three_dimension)
    mass = mass or random.random() * 50 + 0.5
    radius = radius or mass / 10000
    print(radius)
    sim.add(m = mass, r = radius, x = pos[0], y = pos[1], z = pos[2], vx = vel[0], vy = vel[1], vz = vel[2])

def setup(particle_list=[], amount_of_particles=None):
    """
        sets up a sim based on either a list of particles or randomised based on amount
    """
    amount_of_particles = amount_of_particles or len(particle_list)
    sim = rebound.Simulation()
    sim.integrator = "SABA(10,6,4)"
    sim.collision = "direct"
    sim.collision_resolve = "merge"  # "hardsphere"
    sim.dt = 0.001
    sim.units = ("kg", "km", "yr")
    sim.additional_forces = add_thrust

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

def normalize(vec):
    return  vec / np.linalg.norm(vec)

def get_agent_gravity(agent_pos, sim):
    agent_acc =  np.array( (0, 0, 0) )

    for i in range(1, len(sim.particles)):
        particle = sim.particles[i]
        distance = np.array( (particle.x, particle.y, particle.z) ) - agent_pos 
        agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
    
    return agent_acc * sim.G

def get_acceleration(agent_pos, agent_gravity):
    dist_to_target = normalize( np.array(TARGET_POS) - agent_pos ) 

    # agent_acc =  np.array( (agent.ax, agent.ay, agent.az) )
    # print(f"ACCELERATION: {agent_gravity}")

    dot = np.dot( agent_gravity , dist_to_target )
    cross = np.linalg.norm( np.cross( agent_gravity , dist_to_target ) )

    num = MAX_ACCELERATION**2 - cross**2
    # print(f"NUM: {num}")

    if num < 0:
        return agent_gravity

    return (dot + math.sqrt( num )) * dist_to_target - agent_gravity

def add_thrust(simulation):
    sim = simulation.contents
    agent = sim.particles[0]

    agent_pos = np.array( (agent.x, agent.y, agent.z) )
    agent_gravity = get_agent_gravity(agent_pos, sim)
    agent_acc = get_acceleration(agent_pos, agent_gravity)
    agent.ax += agent_acc[0]
    agent.ay += agent_acc[1]
    agent.az += agent_acc[2]


#######################################################################################################

if __name__ == "__main__":

    agent = {
        "mass": 500,
        "pos": (0, 0, 0),
        "radius": 0.01,
        "vel": (-10, 30, 0), #x, y, z
    }

    planets = [
        {
            "mass": 100000000,
            "pos": (30, 20, 0),
            "radius": 5,
            "vel": (0, 0, 0),
        },
    ]

    TARGET_POS = (100, 100, 0)
    MAX_ACCELERATION = 100

    particles = [
        # {"pos":(0,0,0),"vel": (0,0,0), "mass": 0, "radius":0.1},    #Origo
        # {"pos":(1,1.5,0),"vel": (-0.1,0,0), "mass":10, "radius":0.1}, #body1
        # {"pos":(-1,1.7,0),"vel": (0.1,0,0), "mass":10, "radius":0.1}, #body2
        agent,
        planets[0]
    ]

    sim = setup(particle_list=particles)
    # sim = setup(amount_of_particles=10)
    # sim = rebound.SimulationArchive("archive.bin")[0]

    # print all particles 
    # [print(particle) for particle in sim.particles]

    particle_plot = [[] for _ in sim.particles]    
    time = []

    for i in range(1000):

        for j in [i for i in range(len(sim.particles))]:
            particle_plot[j].append((sim.particles[j].x, sim.particles[j].y))
        time.append(sim.t)
        sim.integrate(i*0.1)


    plotter = Plotter()
    # plotter.plot_2d(particle_plot, sim)
    plotter.plot_3d(particle_plot, time)
    plotter.show_plots()
