import numpy as np

def vector_dist(a, b):
    return np.linalg.norm(a-b)

def normalize(vec):
    return  vec / np.linalg.norm(vec)