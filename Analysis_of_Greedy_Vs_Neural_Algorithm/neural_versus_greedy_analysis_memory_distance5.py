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



nnmem_steps = read_datafile('neural_network_steps_taken_diff2cutoff30_diffmem')
nnmem_times = read_datafile('neural_network_time_taken_diff2cutoff30_diffmem')
nnmem_counts = read_datafile('neural_network_counts_taken_diff2cutoff30_diffmem')




distnnmem = np.arange(1,10,1)
datasteps = [nnmem_steps]
datacounts = [nnmem_counts]
datatimes = [nnmem_times]

dist = [distnnmem]
colors = ['green']
labels = ['NN']

plt.subplot(1, 3, 1)
for i in range(0,1):
    mean_data = []
    std_data = []

    for j in range(0,len(dist[i])):
        mean_data.append(np.mean(datasteps[i][j]))
        std_data.append(np.std(datasteps[i][j]))
    plt.plot(dist[i], mean_data, color=colors[i])
    plt.errorbar(dist[i], mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
plt.legend()
plt.xlabel('New particles in activation array per step')
plt.ylabel('Steps to Moved By Cell')
plt.title('Steps')

plt.subplot(1, 3, 2)
for i in range(0,1):
    mean_data = []
    std_data = []

    for j in range(0,len(dist[i])):
        mean_data.append(np.mean(datacounts[i][j]))
        std_data.append(np.std(datacounts[i][j]))
    plt.plot(dist[i], mean_data, color=colors[i])
    plt.errorbar(dist[i], mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
plt.legend()
plt.xlabel('New particles in activation array per step')
plt.ylabel('Que Molecules Hitting Cell')
plt.title('Counts')


plt.subplot(1, 3, 3)
for i in range(0,1):
    mean_data = []
    std_data = []

    for j in range(0,len(dist[i])):
        mean_data.append(np.mean(datatimes[i][j]))
        std_data.append(np.std(datatimes[i][j]))
    plt.plot(dist[i], mean_data, color=colors[i])
    plt.errorbar(dist[i], mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
plt.legend()
plt.xlabel('New particles in activation array per step')
plt.ylabel('Time Taken to Find Source')
plt.title('Time')
plt.tight_layout()
plt.show()




