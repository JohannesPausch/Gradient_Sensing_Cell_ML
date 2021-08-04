from ReceptorNeuralNetwork import *
import numpy as np

X = np.array() #this is a  no. samples x no. receptors array
    
Y = np.array() #this is the location of the receptor closest to each source using nearest_neighbour
    
num_rows, num_cols = np.shape(X)
indices = np.random.choice(range(num_rows), num_rows, replace=False) #if data is ordered use this to randomise it so every source position is trained on
n_train = int(len(indices)/2)
    
training_x = np.array(X)[indices[:n_train].astype(int)] #use first half of the data to train the network
training_y = np.array(Y)[indices[:n_train].astype(int)]
mlp = fit(training_x, training_y)

predict_x = np.array(X)[indices[n_train:].astype(int)] #use second half of the data to predict if network is working
predict_y = np.array(Y)[indices[n_train:].astype(int)]
pred = predict(mlp, predict_x)
print(predict_y, pred)
    