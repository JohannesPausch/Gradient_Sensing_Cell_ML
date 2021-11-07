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

neuraldatadiff1 = read_datafile('neural_network_steps_datacutoff20diffusion1')
greedydatadiff1 = read_datafile('greedy_algorithm_steps_datacutoff20diffusion1')

neuraldatadiff2 = read_datafile('neural_network_steps_datacutoff20diffusion2')
greedydatadiff2 = read_datafile('greedy_algorithm_steps_datacutoff20diffusion2')

dist = np.arange(2,14,1)

data = [neuraldatadiff1, greedydatadiff1, neuraldatadiff2, greedydatadiff2]
colors = ['black', 'blue', 'green', 'red']
labels = ['NN diff=1', 'Greedy diff=1', 'NN diff=2', 'Greedy diff=2']
i=0
for data_set in data:

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
plt.ylabel('No. of cue particles hit cell')
plt.grid()
plt.show()
        
