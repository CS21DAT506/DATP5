from sim_setup.setup import get_vector_with_circular_bound 
from utils.vectors import vector_dist
from settings.settings_access import settings
import numpy as np

def get_valid_environment():
    particles = None
    target_pos = None
    is_valid_conf = False
   
    while not is_valid_conf:
        particles, target_pos = get_environment(settings.num_of_planets)
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

def get_relative_pos():
    return get_vector_with_circular_bound(settings.max_pos_radius)

def get_agent( use_random_pos=False ):
    if not use_random_pos:
        pos = np.array( (0, 0, 0) )
    else:
        pos = get_vector_with_circular_bound(settings.max_pos_radius)

    return get_particle(mass = settings.agent_mass,
                        pos = pos,
                        radius = settings.agent_radius,
                        vel = get_vector_with_circular_bound(settings.max_vel_radius))

def get_mass(average_mass):
    return max(np.random.normal(average_mass, average_mass / 5), 1)

def get_planet(relative_pos):
    return get_particle(mass = get_mass(settings.planets_mass),
                        pos = get_vector_with_circular_bound(settings.max_pos_radius) - relative_pos,
                        radius = settings.planets_radius,
                        vel = get_vector_with_circular_bound(settings.max_vel_radius))

def get_environment(num_of_planets):
    particles = [get_agent()]

    relative_pos = get_relative_pos()

    for _ in range(num_of_planets):
        particles.append( get_planet(relative_pos) )

    return particles, numpy_to_list(get_target_pos(relative_pos))

def get_particle(mass, pos, radius, vel):
    return {
        "mass": mass,
        "pos": numpy_to_list(pos),
        "radius": radius,
        "vel": numpy_to_list(vel)
    }

def numpy_to_list(vec):
    return vec.tolist()

def get_colliding_particles():
    agent = get_particle(1.0, np.array((500, 1000, 0)), 5, np.array((0, 0, 0))  )
    planet_1 = get_particle(500, np.array((1000, 0, 0)), 5, np.array((-100, 0, 0))  )
    planet_2 = get_particle(500, np.array((0, 0, 0)), 5, np.array((100, 0, 0))  )

    return [agent, planet_1, planet_2]

def get_target_pos(relative_pos):
    return get_vector_with_circular_bound(settings.max_pos_radius) - relative_pos

