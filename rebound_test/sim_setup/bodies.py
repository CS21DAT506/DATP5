from sim_setup.setup import get_vector_with_circular_bound 
from utils.vectors import vector_dist
from settings.settings_access import settings
import numpy as np

def gen_valid_environment():
    particles = None
    target_pos = None
    is_valid_conf = False
   
    while not is_valid_conf:
        particles, target_pos = gen_environment(settings.num_of_planets)
        is_valid_conf = is_valid_configuration(particles[settings.agent_index], particles[settings.agent_index+1:], target_pos, settings.min_dist_to_target)
   
    return {"target_pos": target_pos, "particles": particles}

def is_valid_configuration(agent, planets, target, min_dist_to_target):
    for p1 in planets:
        for p2 in planets:
            if not (p1 is p2):
                if vector_dist(p1["pos"], p2["pos"]) < (p1["radius"] + p2["radius"]) * 2:
                    return False

        if vector_dist(p1["pos"], agent["pos"]) < p1["radius"] * 2:
            return False
        
        if vector_dist(p1["pos"], target) < p1["radius"] * 2:
            return False
    
    return vector_dist(agent["pos"], target) >= min_dist_to_target

def gen_relative_pos():
    return get_vector_with_circular_bound(settings.max_pos_radius)

def gen_agent( use_random_pos=False ):
    if not use_random_pos:
        pos = np.array( (0, 0, 0) )
    else:
        pos = get_vector_with_circular_bound(settings.max_pos_radius)

    return gen_particle(mass = settings.agent_mass,
                        pos = pos,
                        radius = settings.agent_radius,
                        vel = get_vector_with_circular_bound(settings.max_vel_radius))

def gen_mass(average_mass):
    return max(np.random.normal(average_mass, average_mass / 5), 1)

def gen_planet(relative_pos):
    return gen_particle(mass = gen_mass(settings.planets_mass),
                        pos = get_vector_with_circular_bound(settings.max_pos_radius) - relative_pos,
                        radius = settings.planets_radius,
                        vel = get_vector_with_circular_bound(settings.max_vel_radius))

def gen_environment(num_of_planets):
    particles = [gen_agent()]
    relative_pos = gen_relative_pos()

    for _ in range(num_of_planets):
        particles.append( gen_planet(relative_pos) )

    return particles, gen_target_pos(relative_pos).toList()

def gen_particle(mass, pos, radius, vel):
    return {
        "mass": mass,
        "pos": pos.toList(),
        "radius": radius,
        "vel": vel.toList()
    }

def gen_target_pos(relative_pos):
    return get_vector_with_circular_bound(settings.max_pos_radius) - relative_pos
