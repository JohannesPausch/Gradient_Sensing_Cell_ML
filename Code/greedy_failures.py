import numpy as np
import ML_Brownian_Interface as mlbi
import random_3d_rotation as r3dr
from scipy.stats import special_ortho_group
from ReceptorNeuralNetwork import *
from IdealDirection import *
from ReceptorMap import *
from datawriteread import *
from sphericaltransf import *
# initialise the cell (load directions, load mlp, load receptors)
# choose initial distance
receptornum = 10
direction_sphcoords = pick_direction(0,10)
print(len(direction_sphcoords))
radius = 1
v = 0.01
receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)
recepsurface_ratio = 10
rate = 1
diffusion = 2 #ideally 0.1 #1 initially
seeds = np.arange(1,201)
distances = np.arange(6,8)
init_distance = 10
mean_final_counts = []
std_final_counts = []
cutoff = 30
cutoffs = np.arange(30,40,5)

for cutoff in cutoffs:
    failed = 0
    for seed in seeds: 
        #print(seed, init_distance)
        init_pos = np.matmul(special_ortho_group.rvs(3,1,random_state= seed),np.array([init_distance,0,0]))
        sourcex= init_pos[0]
        sourcey= init_pos[1]
        sourcez= init_pos[2]
        max_particles = 100000

        # initalize c setup
        brownian_pipe, received, source = mlbi.init_BrownianParticle(sourcex,sourcey,sourcez,rate,diffusion,radius,seed,cutoff)
        ind_list = []
        countparticle = 0
        steps = 0
        count = 1 #count how many particles have been detected so far
        moving = 0
        t = []
        while(count <= max_particles):
            theta_mol = received[0]
            phi_mol = received[1]
            t.append(received[2])
            if count==1: tm = t[0]
            else: tm = t[count-1]-t[count-2]
            ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
            if ind == -1: 
                if moving ==0: 
                    received,source = mlbi.update_BrownianParticle(brownian_pipe)
                else: 
                    next_pos=np.sqrt((sourcex-((tm * v)*np.cos(direction_sphcoords[indm][1])*np.sin(direction_sphcoords[indm][0])))**2+ (sourcey-((tm * v)*np.sin(direction_sphcoords[indm][0])*np.sin(direction_sphcoords[indm][1])))**2  +(sourcez-((tm * v)*np.cos(direction_sphcoords[indm][0])))**2)
                    if next_pos >= cutoff : 
                        failed += 1
                        break
                    received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[indm][0],direction_sphcoords[indm][1], step_radius=tm * v)
                    
                    if str(source[0]) == 'F':
                        #print('# Source found')
                        break
                    else:
                        x,y,z = spherical2cart_point(source[1],source[2])
                        sourcex = source[0]* x
                        sourcey = source[0]* y
                        sourcez = source[0]* z
                        # print(count, sourcex, sourcey, sourcez)

                        steps+=1
            else: #same index for receptors and directions
                
                moving =1
                indm = ind[0][0]
                next_pos=np.sqrt((sourcex-((tm * v)*np.cos(direction_sphcoords[indm][1])*np.sin(direction_sphcoords[indm][0])))**2+ (sourcey-((tm * v)*np.sin(direction_sphcoords[indm][0])*np.sin(direction_sphcoords[indm][1])))**2  +(sourcez-((tm * v)*np.cos(direction_sphcoords[indm][0])))**2)
                # print('prediction to match source',(sourcex-((tm * v)*np.cos(direction_sphcoords[indm][1])*np.sin(direction_sphcoords[indm][0]))), (sourcey-((tm * v)*np.sin(direction_sphcoords[indm][0])*np.sin(direction_sphcoords[indm][1]))),  (sourcez-((tm * v)*np.cos(direction_sphcoords[indm][0]))))
                if next_pos >= cutoff: 
                        failed += 1
                        break
                
                received,source = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[indm][0],direction_sphcoords[indm][1], step_radius=tm * v)
                if str(source[0]) == 'F':
                    #print('# Source found')
                    break
                else:
                    x,y,z = spherical2cart_point(source[1],source[2])
                    sourcex = source[0]* x
                    sourcey = source[0]* y
                    sourcez = source[0]* z
                    # print(count, sourcex, sourcey, sourcez)
                    steps+=1
            count+=1

    with open("greedy_algorithm_failed_10_testb.dat", "a") as output:
        output.write(str(cutoff)+'\t')
        output.write(str(failed)+'\n')