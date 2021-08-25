import random
import math
#This code was taken from GitHub: https://gist.github.com/dinob0t/9597525
#Uses reference: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

def random_on_sphere_points(r,num,seed=0):
    x = []
    y = []
    z = []
    for i in range(0,num):
        random.seed(seed+i)
        zz =  random.uniform(-r,r)
        random.seed(seed+i)
        phi = random.uniform(0,2*math.pi)
        xx = math.sqrt(r**2 - zz**2)*math.cos(phi)
        yy = math.sqrt(r**2 - zz**2)*math.sin(phi)
        x.append(xx)
        y.append(yy)
        z.append(zz)
    return x,y,z
def regular_on_sphere_points(r,num): #for directions
    x = []
    y = []
    z = []
	#Break out if zero points
    if num==0:
        return x,y,z

    a = 4.0 * math.pi*(r**2.0 / num)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta

    for m in range(0,m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * math.pi * n / m_phi
            xx = r * math.sin(theta) * math.cos(phi)
            yy = r * math.sin(theta) * math.sin(phi)
            zz = r * math.cos(theta)
            x.append(xx)
            y.append(yy)
            z.append(zz)
    return x,y,z

