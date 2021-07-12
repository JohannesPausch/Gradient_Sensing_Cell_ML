import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
import scipy.linalg
 
#from analytical import *
#from CostFunction import *
#from GradDescent import *
#from random_3d_rotation import *
############## DATA origin source ###########################
data = np.loadtxt('/Users/Paula/Gradient_Sensing_Cell_ML-main/BrownianParticle_ref.txt',
				delimiter=' ', 	# String used to separate values
				usecols=[0,1,2,3,4,5,6,7], 	# Specify which columns to read
				dtype=np.double) 		# The type of the resulting array
event=data[:,0]
iteration=data[:,1]
theta=data[:,2]
phi=data[:,3]
z=data[:,4] #x before
y=data[:,5] #y
x=data[:,6] #z
t=data[:,7]
numtrain=len(x)+1


Y=np.zeros((1,2)) #origin source
X=np.transpose(np.concatenate((np.transpose(theta).reshape(1,numtrain-1),np.transpose(phi).reshape(1,numtrain-1))))
####################FUNCTIONS###################################
  
def analyticalreg(x,y): 
    #method to solve for beta analytically
    #(X^tX)^-1X^ty
    
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)

def random_3d_rotation(theta,phi,use_seed=None):   
    try:
        np.random.seed(seed=use_seed)
        y=np.matmul(special_ortho_group.rvs(3),np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)]))#special_ortho_group.rvs(3)
        return np.arccos(y[2]), np.mod(np.arctan2(y[1],y[0]),2*np.pi)
    except ValueError:
        if phi.shape != theta.shape:  print("Error in random_3d_rotation: Shapes of phi and theta don't match")
        else: print("Unknown error in random_3d_rotation")
        return theta,phi


def reshapevalues(theta,phi):
    X = np.empty((2*theta.size), dtype=theta.dtype) 
    X[0::2]=theta
    X[1::2]=phi
    return X

##################### SET UP VARIABLES  ######################
groupsize=40

#sweep :

for i in range(0,numtrain-1):
    if (i+groupsize==numtrain-1):
        break
    else:
        theta=X[i:i+groupsize,0]
        phi=X[i:i+groupsize,1]
        Yr=np.array(random_3d_rotation(Y[0,0],Y[0,1])).reshape(1,2)
        [rtheta,rphi]=random_3d_rotation(theta,phi)
        Xr=reshapevalues(rtheta,rphi).reshape(1,2*groupsize)
        if i==0:
            Xi=reshapevalues(theta,phi).reshape(1,2*groupsize)
            Xtot=np.concatenate((Xi,Xr))
            Ytot= np.concatenate((Y, Yr))
        else: 
            Xtot=np.concatenate((Xtot,Xr))
            Ytot= np.concatenate((Ytot, Yr))
        

##################### regression ############################################

beta= analyticalreg(Xtot, Ytot)


#######################RUN ITERATIONS(Gradient descent)###############################################
"""
numiter = 1000
beta = np.random.rand(4,1)  #weights for theta
alpha = 1e-12#learning rate
J=np.zeros((numiter,1))
for i in range(0,numiter): #repeat until convergence
    print(beta)
    beta= GradDescent(alpha,beta,X,y1)
    J[i,0]= CostFunction(beta,X,y1)
""" 

##########################VISUALS#############################################


