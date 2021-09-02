from mpl_toolkits import mplot3d
from datawriteread import *
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
data = np.array(read_datafile('coordinates5'))
print(data)
ax.scatter3D(data[:,1], data[:,2], data[:,3], cmap='Reds');
plt.show()
