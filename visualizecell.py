from datawriteread import read_datafile
from operator import pos
from vpython import *
from ReceptorMap import *
import math
from IdealDirection import *

radius = 1

mindistance = math.pi/10
scene.center = vector(0,0,0)
axes = [vector(1,0,0), vector(0,1,0), vector(0,0,1)]

scene.caption= "Cell and its receptors"
rs,rc,ar = init_Receptors(1,10,0,0)
x,y,z = regular_on_sphere_points(1, 20-1) #directions
print(len(rc))
print(len(x))

     
c = sphere()
c.pos = vector(0,0,0)
c.radius = radius
c.color = vector(0,0.58,0.69)
for i in range(0,len(rc)):              
    cr = ring(pos=vector(rc[i][0],rc[i][1],rc[i][2]),
        axis=vector(rc[i][0],rc[i][1],rc[i][2]),
        radius=mindistance, thickness=0.008)
    sphere(pos = vector(rc[i][0],rc[i][1],rc[i][2]),
       size = vector(0.05, 0.05, 0.05))
for i in range(0,len(x)):
    sphere(pos = vector(x[i],y[i],z[i]),
       size = vector(0.05, 0.05, 0.05), color=vector(1,0,0))




