from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fit(X, y):
    X = MinMaxScaler().fit_transform(X)
    mlp = MLPClassifier(random_state=0, max_iter=400, solver='sgd', learning_rate='constant',
                        momentum=0, learning_rate_init=0.2)

    mlp.fit(X, y)
    return mlp

def predict(mlp: MLPClassifier, X):
    X = MinMaxScaler().fit_transform(X)
    y = mlp.predict(X)
    return y

def nearest_receptor(source, receptors):
    #input x, y, z of source and receptors
    separation_vector = receptors - source
    distance = (np.linalg.norm(row) for row in separation_vector)
    return np.where(distance == np.amin(distance))
    
    

if __name__ == '__main__':
    
    # need to be able to call ReceptorMap.py to return receptors array and activation_receptor array
    
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
    
    
