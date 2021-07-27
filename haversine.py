import numpy as np
import math
#help from: https://janakiev.com/blog/gps-points-distance-python/

def haversine(R,theta1,phi1,theta2,phi2):
    dphi      = phi2 - phi1
    dtheta    = theta2 - theta1
    a = math.sin(dtheta/2)**2 + \
        math.cos(theta1)*math.cos(theta2)*math.sin(dphi/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

