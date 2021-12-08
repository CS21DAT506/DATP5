import numpy as np

def vector_dist(a, b):
    v1 = np.array(a)
    v2 = np.array(b)
    return np.linalg.norm(v1-v2)

def normalize(vec):
    return  vec / np.linalg.norm(np.array(vec))