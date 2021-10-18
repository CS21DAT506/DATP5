import numpy as np
import math
import random
import json
import time

from gekko_wrapper import Gekko
from gekko_plot import GekkoPlotter

debug = True

def get_vector_with_circular_bound(max_radius):
    r = math.sqrt(random.random()) * max_radius
    a = (2 * random.random() * math.pi)
    x = math.cos(a) * r
    y = math.sin(a) * r
    return np.array([x, y])

def vector_dist(a, b):
    return np.linalg.norm(a-b)

def get_mass(min_exp, max_exp):
    return 10 ** random.uniform(min_exp, max_exp)

def is_valid_configuration(agent, planets, target, min_dist_to_target):
    for p1 in planets:
        for p2 in planets:
            if not (p1 is p2):
                if vector_dist(p1["initial_pos"], p2["initial_pos"]) < (p1["radius"] + p2["radius"]) * 2:
                    return False

        if vector_dist(p1["initial_pos"], agent["initial_pos"]) < p1["radius"] * 2:
            return False
        
        if vector_dist(p1["initial_pos"], target) < p1["radius"] * 2:
            return False
    
    return vector_dist(agent["initial_pos"], target) >= min_dist_to_target

def automatedRun(model, agent, planets):
    valid_configuration_found = False
    while not valid_configuration_found:
        agent["initial_pos"] = get_vector_with_circular_bound(MAX_POS_RADIUS)
        agent["initial_velocity"] = get_vector_with_circular_bound(MAX_V_RADIUS)
        planets[0]["initial_pos"] = get_vector_with_circular_bound(MAX_POS_RADIUS)
        # planets[0]["initial_velocity"] = get_vector_with_circular_bound(max_v_radius)
        planets[0]["mass"] = get_mass(4,6)
        planets[0]["radius"] = planets[0]["mass"] / 20000
        target_pos = get_vector_with_circular_bound(MAX_POS_RADIUS)
        valid_configuration_found = is_valid_configuration(agent, planets, target_pos, 200)

    model.setup(agent, planets, target_pos=target_pos)

    try:
        model.solve(disp=False)
    except Exception as e:
        return (target_pos, False)
    
    return (target_pos, True)

def debugRun(model, agent, planets):
    target_pos = [950, 1000]
    model.setup(agent, planets, target_pos=target_pos)

    try:
        model.solve(disp=True)
    except Exception as e:
        return (target_pos, False)
    return (target_pos, True)

agent = {
    "mass": 500,
    "initial_pos": np.array([0,0]),
    "initial_velocity": np.array([-10,0]),
}

planets = [
    {
        "mass": 1000000,
        "initial_pos": np.array([320,420]),
        "radius": 20,
        "initial_velocity": np.array([0,0]),
    },
]

if __name__ == "__main__":
    failed = 0
    target_pos = [100,100]
    m = None
    if (not debug):
        DATA_SIZE = 1
        MAX_POS_RADIUS = 1000
        MAX_V_RADIUS = 50
        
        for iteration in range(DATA_SIZE):
            start_time = time.time()
            m = Gekko()
            (target_pos, didSucceed) = automatedRun(m, agent, planets)
            if (not didSucceed):
                failed += 1
            print(f"{iteration + 1}: {'Succeed' if didSucceed else 'Failed'} (took: {time.time() - start_time} seconds)")
        print(f"failed: {failed}")
    else:
        m = Gekko()
        (target_pos, didSucceed) = debugRun(m, agent, planets)
        
        print(f"{'Succeed' if didSucceed else 'Failed'} (took: {m.m.options.SOLVETIME} seconds)")
        
    results = None

    with open(m.m.path + r"\results.json") as f:
        results = json.load(f)

    GekkoPlotter(results)
    GekkoPlotter.plot3DGraph(results, planets, target_pos)
    GekkoPlotter.showPlots()

