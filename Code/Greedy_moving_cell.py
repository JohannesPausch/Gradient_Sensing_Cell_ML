from textwrap import indent
import numpy as np
import ML_Brownian_Interface as mlbi
import random_3d_rotation as r3dr
from scipy.stats import special_ortho_group
from ReceptorNeuralNetwork import *
from IdealDirection import *
from ReceptorMap import *
from datawriteread import *
from sphericaltransf import *


receptornum = 10
direction_sphcoords = pick_direction(0,10)
radius = 1
recepsurface_ratio = 10

receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)

init_distance = 3
rate = 1
diffusion = 1 #ideally 0.1
seed = 1
cutoff = 20  
init_pos = np.matmul(special_ortho_group.rvs(3, 1,random_state= seed),np.array([init_distance,0,0]))
sourcex= init_pos[0]
sourcey= init_pos[1]
sourcez= init_pos[2]
#x,y,z =spherical2cart_point(direction_sphcoords[0,1],direction_sphcoords[0,2])
#sourcex= init_distance*x
#sourcey= init_distance*y
#sourcez= init_distance*z
max_particles = 100000
print('# movement simulation of the cell with greedy algorithm')
print('# init_distance = '+str(init_distance))
print('# rate = '+str(rate))
print('# diffusion = '+str(diffusion))
print('# seed = '+str(seed))
print('# cutoff = '+str(cutoff))
print('# init_pos = '+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))
print('# max_particles = '+str(max_particles))

# initalize c setup
brownian_pipe, received, source = mlbi.init_BrownianParticle(sourcex,sourcey,sourcez,rate,diffusion,seed,cutoff)
ind_list = []
countparticle = 0
count = 1 #count how many particles have been detected so far
while(count <= max_particles):
    theta_mol = received[0]
    phi_mol = received[1]
    ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
    if ind == -1: 
        received,source = mlbi.update_BrownianParticle(brownian_pipe)
    else: #same index for receptors and directions
        ind = ind[0][0]
        received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[ind][0],direction_sphcoords[ind][1], step_radius=0.1)
        if str(source[0]) == 'F':
            print('# Source found')
            break
        else:
            x,y,z = spherical2cart_point(source[1],source[2])
            sourcex = source[0]* x
            sourcey = source[0]* y
            sourcez = source[0]* z
            print(str(count)+'\t'+str(sourcex)+'\t'+str(sourcey)+'\t'+str(sourcez))
    count+=1