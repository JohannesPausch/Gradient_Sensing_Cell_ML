import numpy as np
import ML_Brownian_Interface as mlbi
import random_3d_rotation as r3dr
from scipy.stats import special_ortho_group
from ReceptorNeuralNetwork import *
from IdealDirection import *
from ReceptorMap import *
from sklearn.preprocessing import MinMaxScaler
from datawriteread import *


# initialise the cell (load directions, load mlp, load receptors)
# choose initial distance
receptornum = 10
direction_sphcoords = pick_direction(0,10)
radius = 1
recepsurface_ratio = 10

receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)

filename = 'Total_mlp'
init_distance = 5
rate = 1
diffusion = 1 #ideally 0.1
seed = 1
cutoff = 20  
init_pos = np.matmul(special_ortho_group.rvs(3),np.array([init_distance,0,0]))
sourcex= init_pos[0]
sourcey= init_pos[1]
sourcez= init_pos[2]
#sourcex,sourcey,sourcez = spherical2cart_point(direction_sphcoords[0,0],direction_sphcoords[0,1])
#sourcex *= init_distance 
#sourcey *= init_distance 
#sourcez *= init_distance 
particlenum = 100
max_particles = 100000
mlp = load_neural_network(filename)
print('# movement simulation of the cell with previously learnt neural network')
print('# init_distance = '+str(init_distance))
print('# rate = '+str(rate))
print('# diffusion = '+str(diffusion))
print('# seed = '+str(seed))
print('# cutoff = '+str(cutoff))
print('# init_pos = '+str(sourcex)+str(sourcey)+str(sourcez))
print('# particlenum = '+str(particlenum))
print('# max_particles = '+str(max_particles))
print('# filename of neural network = '+filename)

# initalize c setup
brownian_pipe, received, source = mlbi.init_BrownianParticle(sourcex,sourcey,sourcez,rate,diffusion,seed,cutoff)

#ind_list = list(-np.ones(particlenum))
ind_list = []
countparticle = 0
count = 1 #count how many particles have been detected so far
while(count <= max_particles):
    theta_mol = received[0]
    phi_mol = received[1]
    ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
    if ind == -1: 
        countparticle +=1
        ind_list.append(-1)
        if countparticle>particlenum:
            ind_list.pop(0) 
    else:
        countparticle +=1
        ind_list.append(ind[0][0])
        if countparticle>particlenum:
            ind_list.pop(0)
    if len(ind_list) == particlenum:
        activation_array = np.zeros((1,len(receptor_sphcoords)))
        for i in ind_list:
            if i != -1:
                activation_array[0,i]+=1
        scaler = MinMaxScaler()
        move = mlp.predict(activation_array)
        if (move == 0).all():
            received,source = mlbi.update_BrownianParticle(brownian_pipe)
        else:
            move= list(move[0])
            received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[move.index(1)][0],direction_sphcoords[move.index(1)][1], step_radius=0.1)
        if str(source[0]) == 'F':
            print('# Source found')
            print(str(count)+'\t'+str(0)+'\t'+str(0)+'\t'+str(0))
            break
        else:
            print(str(count)+'\t'+str(source[0])+'\t'+str(source[1])+'\t'+str(source[2]))

    else: pass
    count+=1
