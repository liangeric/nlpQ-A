import numpy as np 


def diceSim(u,v):
    denom = np.sum(v) + np.sum(u)
    minSum = 0
    for i in range(len(u)):
        minSum += min([u[i], v[i]])
    return (2 * minSum) / denom

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Should be vectorized eventually. this is kinda a zzz 
# using 1 - jaccardSim bc it's weird like that (:
def jaccardSim(u,v):
    minSum, maxSum = 0, 0
    for i in range(len(u)):
        minSum += min([u[i], v[i]])
        maxSum += max([u[i], v[i]])
    return 1 - minSum/maxSum
