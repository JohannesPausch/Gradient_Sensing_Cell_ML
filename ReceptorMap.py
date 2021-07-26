import random
import math
from pprint import pprint
from pointssphere import *
import matplotlib.pyplot as plt
#This code was taken from GitHub: https://gist.github.com/dinob0t/9597525
#Uses reference: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

radius = 1
points = 50

#print("Randomly distributed points")
#x,y,z = random_on_sphere_points(radius,points)

#fig = plt.figure(figsize=(4,4))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x,y,z) 

#plt.show()

print("Evenly distributed points")
x,y,z = regular_on_sphere_points(radius,points)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z) 

plt.show()