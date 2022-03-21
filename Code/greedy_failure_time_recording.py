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
#print(len(direction_sphcoords))
radius = 1
receptor_sphcoords,receptor_cartcoords, activation_array = init_Receptors(radius,receptornum,0)
recepsurface_ratio = 10
rate = 1
diffusion = 2 #ideally 0.1 #1 initially
seeds = np.arange(0,200)
distances = np.arange(6,8)
init_distance = 5
mean_final_counts = []
std_final_counts = []
cutoff = 30
velocities = np.linspace(0.01,0.10,10)
# velocity = 0.1

for velocity in velocities:
    failed = 0
    times = []
    for seed in seeds: 
        open("record_trajectory.txt", "w").close()
        print(seed, velocity)
        init_pos = np.matmul(special_ortho_group.rvs(3,1,random_state= seed),np.array([init_distance,0,0]))
        sourcex= init_pos[0]
        sourcey= init_pos[1]
        sourcez= init_pos[2]
        max_particles = 100000

        # initalize c setup
        brownian_pipe, received, source = mlbi.init_BrownianParticle(sourcex,sourcey,sourcez,rate,diffusion,radius,seed,cutoff,None,None,'record_trajectory.txt')
        #print('Pipe initiliased')
        ind_list = []
        countparticle = 0
        steps = 0
        count = 1 #count how many particles have been detected so far
        moving = 0
        
        
        while(count <= max_particles):
            #print(count)
            try:
                theta_mol = received[0]
                phi_mol = received[1]
                # t.append(received[2])
            except:
                print(received)
            ind = activation_Receptors(theta_mol,phi_mol,receptor_sphcoords,radius,radius*math.pi/recepsurface_ratio)
            if ind == -1: 
                if moving ==0: # this is a dummy value so that program always goes into else-option 
                    received = mlbi.update_BrownianParticle(brownian_pipe)
                    #print('# cell does have direction yet')
                else: 
                    #next_pos=np.sqrt((sourcex-((tm * v)*np.cos(direction_sphcoords[indm][1])*np.sin(direction_sphcoords[indm][0])))**2+ (sourcey-((tm * v)*np.sin(direction_sphcoords[indm][0])*np.sin(direction_sphcoords[indm][1])))**2  +(sourcez-((tm * v)*np.cos(direction_sphcoords[indm][0])))**2)
                    #if next_pos >= cutoff : 
                    #    failed += 1
                    #    break
                    received = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[indm][0],direction_sphcoords[indm][1],velocity,True)
                    #print('# cell uses previous direction')
                    if str(received) == 'SOURCE FOUND':
                        # print('# Source found')
                        with open('record_trajectory.txt', 'r') as f:
                            last_line = f.readlines()[-3]
                        last_line=last_line.strip('\n')
                        time,sourcex,sourcey,sourcez=last_line.split()
                        total_time = float(time)+(np.sqrt(float(sourcex)**2+float(sourcey)**2+float(sourcez)**2)-2)/velocity
                        times.append(total_time)

                        break
                    if str(received) == 'Left range':
                        #print('# Cell left range')
                        failed += 1
                        break
            else: #same index for receptors and directions
                
                moving =1
                indm = ind[0][0]
                #next_pos=np.sqrt((sourcex-((tm * v)*np.cos(direction_sphcoords[indm][1])*np.sin(direction_sphcoords[indm][0])))**2+ (sourcey-((tm * v)*np.sin(direction_sphcoords[indm][0])*np.sin(direction_sphcoords[indm][1])))**2  +(sourcez-((tm * v)*np.cos(direction_sphcoords[indm][0])))**2)
                # print('prediction to match source',(sourcex-((tm * v)*np.cos(direction_sphcoords[indm][1])*np.sin(direction_sphcoords[indm][0]))), (sourcey-((tm * v)*np.sin(direction_sphcoords[indm][0])*np.sin(direction_sphcoords[indm][1]))),  (sourcez-((tm * v)*np.cos(direction_sphcoords[indm][0]))))
                #if next_pos >= cutoff: 
                #        failed += 1
                #        break
                received = mlbi.update_BrownianParticle(brownian_pipe,direction_sphcoords[indm][0],direction_sphcoords[indm][1], velocity,True)
                #print('# cell has new direction')
                if str(received) == 'SOURCE FOUND':
                    # print('# Source found')
                    with open('record_trajectory.txt', 'r') as f:
                            last_line = f.readlines()[-3]
                    last_line=last_line.strip('\n')
                    time,sourcex,sourcey,sourcez=last_line.split()
                    total_time = float(time)+(np.sqrt(float(sourcex)**2+float(sourcey)**2+float(sourcez)**2)-2)/velocity
                    times.append(total_time)
                    break
                if str(received) == 'Left range':
                    #print('# Cell Left range')
                    failed += 1
                    break
            count+=1

    with open("greedy_algorithm_D_2_R_1_talia.dat", "a") as output:
        output.write(str(velocity)+'\t')
        output.write(str(failed)+'\t')
        for idx, tm in enumerate(times):
            if idx < len(times)-1:
                output.write(str(tm)+'\t')
            else:
                output.write(str(tm)+'\n')