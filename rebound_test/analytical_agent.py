import math
import numpy as np
from constants import *

def normalize(vec):
    return  vec / np.linalg.norm(vec)

def get_agent_gravity(agent_pos, sim):
    agent_acc =  np.array( (0, 0, 0) )

    for i in range(1, len(sim.particles)):
        particle = sim.particles[i]
        distance = np.array( (particle.x, particle.y, particle.z) ) - agent_pos 
        agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
    
    return agent_acc * sim.G

PREFERRED_VALUE = 0.5
INVALID_VALUE = -1

def scale_policy(velocity_scale_min, velocity_scale_max):
    if (velocity_scale_max >= PREFERRED_VALUE and velocity_scale_min <= PREFERRED_VALUE):
        return PREFERRED_VALUE
    elif (velocity_scale_max < PREFERRED_VALUE and velocity_scale_min < PREFERRED_VALUE):
        if (velocity_scale_max >= 0):
            return velocity_scale_max
    elif (velocity_scale_max > PREFERRED_VALUE and velocity_scale_min > PREFERRED_VALUE):
        if (velocity_scale_min <= 1):
            return velocity_scale_min
    else:
        return INVALID_VALUE

def get_velocity_scale(agent_gravity, agent_velocity):
    velocity_length = np.linalg.norm( agent_velocity )
    dot = - np.dot( agent_velocity / velocity_length, agent_gravity  )

    diff = MAX_ACCELERATION**2 + dot**2 - np.linalg.norm( agent_gravity )**2
    if (diff < 0):
        return INVALID_VALUE
    diff = math.sqrt(diff)

    velocity_scale_min = (dot - diff) / velocity_length
    velocity_scale_max = (dot + diff) / velocity_length    

    return scale_policy(velocity_scale_min, velocity_scale_max)

def get_acceleration(agent_pos, agent_velocity, agent_gravity):
    dist_to_target = normalize( np.array(TARGET_POS) - agent_pos ) 
    velocity_scale = get_velocity_scale(agent_gravity, agent_velocity)
    if (velocity_scale == INVALID_VALUE):
        return np.array( [0, 0, 0] )
    c = velocity_scale * agent_velocity + agent_gravity

    dot = np.dot( c , dist_to_target )
    cross = np.linalg.norm( np.cross( c , dist_to_target ) )

    num = MAX_ACCELERATION**2 - cross**2

    if num < 0:
        return np.array( [0, 0, 0] )

    return (dot + math.sqrt( num )) * dist_to_target - c

def add_thrust(simulation):
    sim = simulation.contents
    agent = sim.particles[0]

    agent_pos = np.array( (agent.x, agent.y, agent.z) )
    agent_velocity = np.array( (agent.vx, agent.vy, agent.vz) )
    agent_gravity = get_agent_gravity(agent_pos, sim)
    agent_acc = get_acceleration(agent_pos, agent_velocity, agent_gravity)
    agent.ax += agent_acc[0]
    agent.ay += agent_acc[1]
    agent.az += agent_acc[2]