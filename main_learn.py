import numpy as np
from ReceptorMap import *


############## intialize receptor map ###################
radius = 1
receptornum = 10
mindistance = 0.05

receptor_sphcoords,receptor_cartcoords, activation_receptors = init_Receptors(receptornum,radius)
plot = visualize_Receptors(receptor_cartcoords,radius,mindistance)
plot.show()

### action #####
# need to link matlab to python, not sure yet how

#### parameters for brownian particle ########



