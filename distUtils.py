import numpy as np 


def diceSim(u,v):
    denom = np.sum(v) + np.sum(u)
    minSum = 0
    for i in range(len(u)):
        minSum += min([u[i], v[i]])
    return (2 * minSum) / denom

