import numpy as np
import math
def sphericaltransf(x,y,z):
    receptnum = len(x)
    r = np.zeros((receptnum))
    theta = np.zeros((receptnum))
    phi = np.zeros((receptnum))

    for i in range(0,receptnum):
        r[i] = math.pow(x[i],2) + math.pow(y[i],2) + math.pow(z[i],2)
        theta[i] = math.pi/2 - math.acos(z[i]/r[i])
        phi[i] = math.atan2(y[i],x[i])
    return theta,phi
