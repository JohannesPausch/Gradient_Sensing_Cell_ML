import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
import os
import sys

def read_datafile(filename):
    with open(filename+'.txt','r') as fin:
        a = []
        for line in fin:
            if line.startswith('#'): pass
            else:
                b = []
                line = line.split()
                for x in line: 
                    b.append(float(x))
                a.append(b)
    return(a)

neural_steps = read_datafile('NN_actual_steps_cutoff20diff2')
greedy_steps = read_datafile('greedy_actual_stepscutoff20diff2')

neural_counts = read_datafile('neural_network_counts_datacutoff20diffusion2')
greedy_counts = read_datafile('greedy_algorithm_counts_datacutoff20diffusion2')


mean_ratio1 = []
for row, steps in enumerate(neural_steps):
    ratio1=[]
    for counter, value in enumerate(steps):
        ratio = (neural_counts[row][counter]-99)/value
        ratio1.append(ratio)
    mean = np.mean(ratio1)
    mean_ratio1.append(mean)

print(mean_ratio1)


mean_ratio2 = []
for row, steps in enumerate(greedy_steps):
    ratio1=[]
    for counter, value in enumerate(steps):
        ratio = (greedy_counts[row][counter])/value
        ratio1.append(ratio)
    mean = np.mean(ratio1)
    mean_ratio2.append(mean)

print(mean_ratio2)

dist1 = np.arange(2,14,1)

plt.plot(dist1, mean_ratio1, label ='neural')
plt.plot(dist1, mean_ratio2, label='greedy')
plt.xlabel('Distance')
plt.ylabel('No. of particles hitting cell per step')
plt.legend()
plt.show()

