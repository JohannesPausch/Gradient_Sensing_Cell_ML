import numpy as np
import ML_Brownian_Interface as mlbi
import random_3d_rotation as r3dr
from scipy.stats import special_ortho_group
from ReceptorNeuralNetwork import *
from IdealDirection import *
from ReceptorMap import *

# initialise the cell (load directions, load mlp, load receptors)
# choose initial distance
receptornum = 10
direction_sphcoords = pick_direction(0,10)
radius = 1
recepsurface_ratio = 10

receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)

filename = 'Total_mlp'
init_distance = 6
rate = 1
diffusion = 0.1 #ideally 0.1
seed = 1
cutoff = 30
init_pos = np.matmul(special_ortho_group.rvs(3),np.array([init_distance,0,0]))
particlenum = 20
max_particles = 10000
mlp = load_neural_network(filename)
print('# movement simulation of the cell with previously learnt neural network')
print('# init_distance = '+str(init_distance))
print('# rate = '+str(rate))
print('# diffusion = '+str(diffusion))
print('# seed = '+str(seed))
print('# cutoff = '+str(cutoff))
print('# init_pos = '+str(init_pos))
print('# particlenum = '+str(particlenum))
print('# max_particles = '+str(max_particles))
print('# filename of neural network = '+filename)

# initalize c setup
brownian_pipe, received, source = mlbi.init_BrownianParticle(init_pos[0],init_pos[1],init_pos[2],rate,diffusion,seed,cutoff)

activation_array = np.zeros((1,len(receptor_sphcoords)))
#ind_list = list(-np.ones(particlenum))
ind_list = []
countparticle = 0
count = 1 #count how many particles have been detected so far
while(count <= max_particles):
    theta_mol = received[0]
    phi_mol = received[1]
    ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
    print(ind)
    if ind == -1: 
        countparticle +=1
        ind_list.append(-1)
        if countparticle>20:
            ind_list.pop(0) 
    else:
        countparticle +=1
        ind_list.append(ind[0])
        activation_array[0,ind[0]] += 1
        if countparticle>20:
            if ind_list[0]!= -1:
                print('hi')
                activation_array[0,ind_list[0]] -= 1
            if activation_array[0,ind_list[0]] < 0:
                print('# Error: negative activation')
            ind_list.pop(0)
    #if activation_array
    if len(ind_list)==20:
        move = predict(mlp, activation_array)
        print(move)
        print(activation_array)
        if (move == 0).all():
            received,source = mlbi.update_BrownianParticle(brownian_pipe)
        else:
            move= list(move[0])
            received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[move.index(1)][0],direction_sphcoords[move.index(1)][1], step_radius=0.1)
        print(str(count)+'\t'+str(source[0])+'\t'+str(source[1])+'\t'+str(source[2]))
        if str(received[0]) == 'SOURCE':
            print('# Source found')
            break
    count+=1
