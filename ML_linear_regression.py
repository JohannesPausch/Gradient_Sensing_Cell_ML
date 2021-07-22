
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
import scipy.linalg
 
from analyticalreg import *
from reshapevalues import *
from random_3d_rotation import *

############## DATA origin source ###########################
data = np.loadtxt('BrownianParticle_RUNRUN01/BrownianParticle_RUNRUN01_1_1000_P01_20210705_173431.txt',
				delimiter=' ', 	# String used to separate values
				usecols=[0,1,2,3,4,5,6,7], 	# Specify which columns to read
				dtype=np.double) 		# The type of the resulting array
theta_data=data[:,2]
phi_data=data[:,3]
numtrain=len(theta_data)+1

Ydata=np.zeros((1,2)) #origin source
Xdata=np.transpose(np.concatenate((np.transpose(theta_data).reshape(1,numtrain-1),np.transpose(phi_data).reshape(1,numtrain-1))))

##################### SET UP VARIABLES  ######################

# A sweep is done with the window of size "groupsize" that goes down the rows of matrix Xdata with step 1 row.

# Xtot is a matrix of size: (number of groups) x (2*groupsize+1). The 2*groupsize+1 is the column dimension since 
# each row takes all the data of a group and organizes it in the following way using the reshapevalues function: 
# theta_1 phi_1 theta_2 phi_2... theta(2*groupsize) phi_(2*groupsize). In the end a 1 is added as the first variable of each row
# to account for the intercept.

# In Xtot, the row 0 is not rotated and row 1 is the random rotation of row 0,
# the rest of the rows correspond to randomly rotated variables of the selected window of Xdata.

groupsize=40

for i in range(0,numtrain-1):
    if (i+groupsize>numtrain-1):
        print(i+groupsize) #The window is larger than the actual rows left of the training matrix. 
        break
    else:
        theta=Xdata[i:i+groupsize,0]
        phi=Xdata[i:i+groupsize,1]
        Yrotdata=np.array(random_3d_rotation(Ydata[0,0],Ydata[0,1],i)).reshape(1,2)
        [rtheta,rphi]=random_3d_rotation(theta,phi,i)
        Xrotdata=reshapevalues(rtheta,rphi).reshape(1,2*groupsize)
        if i==0:
            Xinit=reshapevalues(theta,phi).reshape(1,2*groupsize)
            Xtot=np.concatenate((Xinit,Xrotdata))
            Ytot= np.concatenate((Ydata, Yrotdata))
        else: 
            Xtot=np.concatenate((Xtot,Xrotdata))
            Ytot= np.concatenate((Ytot, Yrotdata))
        
#add intercept: 
Xtot=np.insert(Xtot, 0, 1, axis=1)
##################### Linear regression ############################################


beta= analyticalreg(Xtot, Ytot)
<<<<<<< HEAD
print(beta[0:10,:])
plt.hist(beta[0::2,0])
plt.savefig('betaHistogramm01.png')
=======