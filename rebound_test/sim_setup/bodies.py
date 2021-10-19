import numpy as np

from settings.settings import MAX_V_RADIUS, MAX_POS_RADIUS
from sim_setup.setup import get_vector_with_circular_bound 

from utils.vectors import vector_dist

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

relative_pos = get_vector_with_circular_bound(MAX_POS_RADIUS)

def get_agent():
    return {
        "mass": 500,
        "pos": np.array( (0, 0, 0) ),
        "radius": 0.01,
        "vel": get_vector_with_circular_bound(MAX_V_RADIUS), 
    }

def get_planet():
    return {
            "mass": 1e10,
            "pos": get_vector_with_circular_bound(MAX_POS_RADIUS) - relative_pos,
            "radius": 5,
            "vel": get_vector_with_circular_bound(MAX_V_RADIUS),
    }

def get_particles(num_of_planets):
    particles = [get_agent()]

    for _ in range(num_of_planets):
        particles.append( get_planet() )

    return particles

def get_target_pos():
    return get_vector_with_circular_bound(MAX_POS_RADIUS) - relative_pos

