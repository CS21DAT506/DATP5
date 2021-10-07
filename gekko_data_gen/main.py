import numpy as np
import math
import random
import json

from gekko_wrapper import Gekko
from gekko_plot import GekkoPlotter

def get_vector_with_circular_bound(max_radius):
    r = math.sqrt(random.random()) * max_radius
    a = (2 * random.random() * math.pi)
    x = math.cos(a) * r
    y = math.sin(a) * r
    return np.array([x, y])


agent = {
    "mass": 500,
    "initial_pos": np.array([0,0]),
    "initial_velocity": np.array([-10,0]),
}

planets = [
    {
        "mass": 100000,
        "initial_pos": np.array([30,20]),
        "radius": 5,
        "initial_velocity": np.array([0,0]),
    },
]


if __name__ == "__main__":
    m = Gekko()
    m.setup(agent, planets)
    # m.solve(disp=False)

    results = None

    # with open(m.m.path+"//results.json") as f:
    #     results = json.load(f)
    with open(r"C:\Users\jakob\AppData\Local\Temp\tmp2vjlcthtgk_model0"+"//results.json") as f:
        results = json.load(f)

    GekkoPlotter(results)

