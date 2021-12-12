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

greedyl_steps = read_datafile('greedy_algorithm_stepsmoved_diff2cutoff30_lazy')
greedyl_times = read_datafile('greedy_algorithm_time_diff2cutoff30_lazy')
greedyl_counts = read_datafile('greedy_algorithm_counts_diff2cutoff30_lazy')

greedy_steps = read_datafile('greedy_steps_taken_diff2_cutoff30_newcode')
greedy_times = read_datafile('greedy_time_diff2_cutoff30_newcode')
greedy_counts = read_datafile('greedy_counts_diff2_cutoff30_newcode')

steps_data = [greedyl_steps, greedy_steps]
counts_data = [greedyl_counts, greedy_counts]
times_data =[greedyl_times, greedy_times]
colors=['blue','green']
labels=['Greedy lazy Algorithm', 'Greedy Algorithm']
dist = np.arange(2,12,1)
i=0
plt.subplot(1, 3, 1)
for data_set in steps_data:
    dist = np.arange(2,len(data_set)+2)
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
plt.ylabel('Steps to Moved By Cell')
plt.title('Steps')

i=0
plt.subplot(1, 3, 2)
for data_set in counts_data:
    dist = np.arange(2,len(data_set)+2)
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
plt.ylabel('Que Molecules Hitting Cell')
plt.title('Counts')

i=0
plt.subplot(1, 3, 3)
for data_set in times_data:
    dist = np.arange(2,len(data_set)+2)
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
plt.ylabel('Time Taken to Find Source')
plt.title('Time')
plt.tight_layout()
plt.show()
