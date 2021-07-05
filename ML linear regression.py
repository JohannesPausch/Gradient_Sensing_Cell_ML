import numpy as np
import matplotlib.pyplot as plt

##############DATA###########################
data = np.loadtxt('BrownianParticle_ref.txt',
				delimiter=' ', 	# String used to separate values
				usecols=[0,1,2,3,4,5,6,7], 	# Specify which columns to read
				dtype=np.double) 		# The type of the resulting array
event=data[:,0]
iteration=data[:,1]
theta=data[:,2]
phi=data[:,3]
x=data[:,4]
y=data[:,5]
z=data[:,6]
t=data[:,7]
m=len(x)+1
#training data for source at origin
y1=source_theta_rel_sphere=np.zeros(m-1) # theta of source
y2=source_phi_rel_sphere=np.zeros(m-1)# phi of source, doesn't matter when source in the x axis
X=np.concatenate((np.ones((1,m-1)),np.transpose(x).reshape(1,m-1),np.transpose(y).reshape(1,m-1),
                  np.transpose(z).reshape(1,m-1),
                  np.transpose(t).reshape(1,m-1)))
####################FUNCTIONS###################################
def CostFunction(beta,X,y): #Cost function
    h = (np.matmul(np.transpose(beta),X)-y)
    m = len(y)
    J = 1/(2*m) * np.sum(np.square(h))
    return J
def GradDescent(alpha,beta,X,y): #Gradient descent
    m = len(y)
    for i in range(len(beta)):
        beta[i,0] = beta[i,0] - alpha * 1/m * (np.matmul((np.matmul(np.transpose(beta),X)-y), np.reshape(X[i,:], (m, 1))))
    return beta    

#######################RUN ITERATIONS###############################################
numiter = 1000
#X =  #matrix of m training examples with parameters: 1, postition, time (steps)
#theta =  # matrix of m angles (2 angles from the data)
beta = np.random.rand(5,1)  #weights for theta
alpha = 1e-12#learning rate
J=np.zeros((numiter,1))
for i in range(0,numiter): #repeat until convergence
    print(beta)
    beta= GradDescent(alpha,beta,X,y1)
    J[i,0]= CostFunction(beta,X,y1)
 
##########################VISUALS#############################################

plt.plot(J[0:3],range(1,4))
plt.show()
