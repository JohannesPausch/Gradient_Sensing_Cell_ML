import random
import math
from pprint import pprint
from pointssphere import *
import matplotlib.pyplot as plt
from haversine import * 
from sphericaltransf import *
####################################
#Distribution of receptors:

#This code was taken from GitHub: https://gist.github.com/dinob0t/9597525
#Uses reference: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf


radius = 1
points = 10

print("Randomly distributed points")
x,y,z = random_on_sphere_points(radius,points)
receptors= np.concatenate(([x],[y],[z])).T
#pprint(x)
#pprint(receptors)
receptornum= len(x)
theta,phi= sphericaltransf(x,y,z)
activation_receptor=np.zeros((receptornum))
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z) 

plt.show()

plt.savefig('ReceptorMapOutput01.png')

"""
print("Evenly distributed points")
x,y,z = regular_on_sphere_points(radius,points)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z) 

<<<<<<< HEAD
plt.show()
"""
##############################################
#Check whether a recetor is close to the molecule
#activate closest receptor
#need data of where the molecule hits
"""
MinDistance=0.05

for i in range(0,receptornum):
    finaldistance = 100
    distance = haversine(radius,molecule_theta,molecule_phi,theta[i],phi[i])
    if distance < MinDistance and distance < finaldistance:
        finaldistance = distance
        finalindex = i

activation_receptor[finalindex] += 1
"""
####################################


