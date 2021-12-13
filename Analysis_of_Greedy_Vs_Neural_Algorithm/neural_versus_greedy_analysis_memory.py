import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
import os
import sys

def read_datafile(filename):
    with open(filename+".txt","r") as fin:
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

nn_steps = read_datafile('neural_network_steps_taken_diff2_cutoff30_newcode')
nn_times = read_datafile('neural_network_time_diff2_cutoff30_newcode')
nn_counts = read_datafile('neural_network_counts_diff2_cutoff30_newcode')

nnmem20_steps = read_datafile('neural_network_steps_taken_diff2cutoff30_mem20')
nnmem20_times = read_datafile('neural_network_time_taken_diff2cutoff30_mem20')
nnmem20_counts = read_datafile('neural_network_counts_taken_diff2cutoff30_mem20')

nnmem50_steps = read_datafile('neural_network_steps_taken_diff2cutoff30_mem50')
nnmem50_times = read_datafile('neural_network_time_taken_diff2cutoff30_mem50')
nnmem50_counts = read_datafile('neural_network_counts_taken_diff2cutoff30_mem50')

greedy_steps = read_datafile('greedy_algorithm_stepsmoved_diff2cutoff30_lazy')
greedy_times = read_datafile('greedy_algorithm_time_diff2cutoff30_lazy')
greedy_counts = read_datafile('greedy_algorithm_counts_diff2cutoff30_lazy')


distnnmem = np.arange(2,12,1)
distnn = np.arange(2,21,1)
datasteps = [nn_steps, greedy_steps, nnmem20_steps,nnmem50_steps]
datacounts = [nn_counts, greedy_counts,nnmem20_counts,nnmem50_counts]
datatimes = [nn_times, greedy_times, nnmem20_times,nnmem50_times]

dist = [distnn,distnnmem, distnnmem,distnnmem]
colors = ['green', 'black', 'blue','red']
labels = ['NN 1 new','greedy','NN 20 new', 'NN 50 new']

plt.subplot(1, 3, 1)
for i in range(0,4):
    mean_data = []
    std_data = []

    for j in range(0,len(dist[i])):
        mean_data.append(np.mean(datasteps[i][j]))
        std_data.append(np.std(datasteps[i][j]))
    plt.plot(dist[i], mean_data, color=colors[i])
    plt.errorbar(dist[i], mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
plt.legend()
plt.xlabel('Initial distance of Cell from Source')
plt.ylabel('Steps to Moved By Cell')
plt.title('Steps')

plt.subplot(1, 3, 2)
for i in range(0,4):
    mean_data = []
    std_data = []

    for j in range(0,len(dist[i])):
        mean_data.append(np.mean(datacounts[i][j]))
        std_data.append(np.std(datacounts[i][j]))
    plt.plot(dist[i], mean_data, color=colors[i])
    plt.errorbar(dist[i], mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
plt.legend()
plt.xlabel('Initial distance of Cell from Source')
plt.ylabel('Que Molecules Hitting Cell')
plt.title('Counts')


plt.subplot(1, 3, 3)
for i in range(0,4):
    mean_data = []
    std_data = []

    for j in range(0,len(dist[i])):
        mean_data.append(np.mean(datatimes[i][j]))
        std_data.append(np.std(datatimes[i][j]))
    plt.plot(dist[i], mean_data, color=colors[i])
    plt.errorbar(dist[i], mean_data, yerr=std_data, color=colors[i],fmt='o',  markersize=6, capsize=5, label=labels[i])
plt.legend()
plt.xlabel('Initial distance of Cell from Source')
plt.ylabel('Time Taken to Find Source')
plt.title('Time')
plt.tight_layout()
plt.show()




