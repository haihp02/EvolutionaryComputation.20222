import numpy as np
from data import Data
from copy import deepcopy

def is_within_FoV(bisector, target, sensor, radius, q):
    target = np.asarray(target)
    sensor = np.asarray(sensor)
    bisector = np.asarray(bisector)
    v = target - sensor
    dist = np.linalg.norm(v)
    scalar = bisector.dot(v)
    return scalar + 1e-7 >= radius*dist*np.cos(np.pi/q) and dist - 1e-7 <= radius

def init_T(data: Data):
    sensors = data.sensors
    targets = data.targets
    radius = data.radius
    n = data.n
    m = data.m
    q = data.q
    T = np.zeros((n, m, q), dtype=bool)

    bisectors = []
    for i in range(q):
        bisectors.append((radius*np.cos(np.pi*(1 + i*2)/q), radius*np.sin(np.pi*(1 + i*2)/q)))

    for i in range(n):
        for j in range(m):
            for p in range(q):
                T[i, j, p] = is_within_FoV(bisector=bisectors[p], target=targets[j], sensor=sensors[i], radius=radius, q=q)

    return T