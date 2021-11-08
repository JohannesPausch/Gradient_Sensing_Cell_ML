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

neuraldatadiff1 = read_datafile('NN_actual_steps_cutoff20diff2')
greedydatadiff1 = read_datafile('greedy_actual_stepscutoff20diff2')

neuraldatadiff2 = read_datafile('neural_network_counts_datacutoff20diffusion2')
greedydatadiff2 = read_datafile('greedy_algorithm_counts_datacutoff20diffusion2')

dist = np.arange(2,14,1)

data1 = [neuraldatadiff1, greedydatadiff1]
data2 = [neuraldatadiff2, greedydatadiff2]
colors = ['red', 'black']
labels = ['Neural Network', 'Greedy Algorithm']
i=0

plt.subplot(1, 2, 1)
for data_set in data1:

    mean_data = []
    std_data = []
    for j in range(0,len(dist)):
        mean_data.append(np.mean(data_set[j]))
        std_data.append(np.std(data_set[j]))
    plt.plot(dist, mean_data, color=colors[i])
    plt.errorbar(dist, mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
    i+=1
plt.legend()
plt.xlabel('Initial distance of Cell from Source')
plt.ylabel('Steps to Source')
plt.title('Steps Cell Makes')
plt.grid()

i=0

plt.subplot(1, 2, 2)
for data_set in data2:

    mean_data = []
    std_data = []
    for j in range(0,len(dist)):
        mean_data.append(np.mean(data_set[j]))
        std_data.append(np.std(data_set[j]))
    plt.plot(dist, mean_data, color=colors[i])
    plt.errorbar(dist, mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
    i+=1
plt.legend()
plt.xlabel('Initial distance of Cell from Source')
plt.ylabel('Cue Particle Hits on Cell')
plt.title('No. of Particles hitting Cell')
plt.grid()
plt.show()

