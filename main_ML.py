from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
import numpy as np
import pickle
"""
######## data ##################
#direction picking for cell: 
#create some kind of print command for user to select the directions

#create data
X, Y = datacreate(direction_sphcoords, receptornum=10,particlenum=20)

print(X)
print(Y)
pickle.dump(X, open('X', 'wb'))
pickle.dump(Y, open('Y', 'wb')) 
"""
direction_sphcoords = pick_direction(0, 10)


#X = pickle.load(open('X', 'rb'))
#Y = pickle.load(open('Y', 'rb'))
#print(X.shape)
#print(Y.shape)
#print(Y)
#Y = Y.T #if not, doesn't work with separate train set

#X1, Y1 = datacreate(direction_sphcoords, receptornum=10,particlenum=20, recepsurface_ratio=10, radiusexact=1,diffusionexact=0.5,rateexact=0.1)
#pickle.dump(X1, open('X1', 'wb'))
#pickle.dump(Y1, open('Y1', 'wb')) 
#X1 = pickle.load(open('X1', 'rb'))
#Y1 = pickle.load(open('Y1', 'rb'))
#print(Y1)
#print(X1)
#training_x, training_y, predict_x, predict_y = separate_train_set(X1, Y1)

training_x, training_y, predict_x, predict_y = separate_train_set(X, Y)
mlp = train(training_x, training_y, layers_tuple = (6,4), max_iterations=1000)
filename = save_neural_network(mlp)
restored_mlp = load_neural_network(filename)
print(filename)

acc, directprob = test(restored_mlp, predict_x, predict_y)
