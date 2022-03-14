import matplotlib.pyplot as plt
import numpy as np 

fig = plt.figure()
ax = plt.subplot(111)


distance_5_v_01 = np.loadtxt('greedy_algorithm_failed_5_talia.dat',unpack=True)
distance_15_v_01 = np.loadtxt('greedy_algorithm_failed_15_talia.dat',unpack=True)

cutoffs = np.arange(20,52,2)

init_distances = [5,15]

ax.plot(cutoffs, distance_5_v_01[1],label='Distance=5, Velocity=0.01', color='blue')
ax.plot(cutoffs, distance_15_v_01[1],label='Distance=15, Velocity=0.01',color='darkblue')

ax.set_xlabel('Cutoff')
ax.set_ylabel('Number of Failures')


distance_10_v_055 = np.loadtxt('greedy_algorithm_failed_10_v0.055.dat',unpack=True)
distance_10_v_055_2 = np.loadtxt('greedy_algorithm_failed_10_v0.055_2.dat',unpack=True)

distance_5_v_055 = np.loadtxt('greedy_algorithm_failed_5_v0.055.dat',unpack=True)
distance_5_v_055_2 = np.loadtxt('greedy_algorithm_failed_5_v0.055_2.dat',unpack=True)

distance_15_v_055 = np.loadtxt('greedy_algorithm_failed_15_v0.055_2.dat',unpack=True)

ax.errorbar(cutoffs, np.mean([distance_10_v_055[1],distance_10_v_055_2[1]],axis=0), yerr=np.std([distance_10_v_055[1],distance_10_v_055_2[1]],axis=0), label='Distance=10, Velocity=0.055', color='indianred')
ax.errorbar(cutoffs, np.mean([distance_5_v_055[1],distance_5_v_055_2[1]],axis=0), yerr=np.std([distance_5_v_055[1],distance_5_v_055_2[1]],axis=0), label='Distance=5, Velocity=0.055',color='red')
print(distance_15_v_055)
ax.plot(cutoffs[:len(distance_15_v_055[1])], distance_15_v_055[1], label='Distance=15, Velocity=0.055',color='darkred')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
